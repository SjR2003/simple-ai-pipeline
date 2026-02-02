import shutil
import datetime
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from core.base_task import BaseTask
from core.task_registry import register_task
from tasks.numeric.task_preprocess_ds.config import PreprocessDsConfig
from tasks.numeric.task_preprocess_ds.result_schema import PreprocessResult
from tasks.numeric.task_load_ds.result_schema import LoadDsResult

warnings.filterwarnings("ignore")

current_state = datetime.datetime.now()
saved_state = None


@register_task("preprocess_ds")
class PreprocessDs(BaseTask):
    def __init__(self, config: dict, input_data: any):
        super().__init__(config, input_data)

        global current_state, saved_state
        if saved_state == None:
            saved_state = current_state
            self._new_id = True
        elif current_state == saved_state:
            self._new_id = False

        self._output_path = self._output_path / Path(__file__).resolve().parent.name
        if self._output_path.exists() and self._new_id:
            shutil.rmtree(self._output_path)

        self._output_path.mkdir(exist_ok=True, parents=True)

        self._run_id = 0
        result_dir = f"result_{self._run_id}"

        if self._output_path.exists() and not self._new_id:
            self._run_id = self._get_last_folder()
            result_dir = f"result_{self._run_id}"

        self._output_path = self._output_path / result_dir
        self._output_path.mkdir(exist_ok=True, parents=True)

    def _validate_config(self) -> None:
        self._config = PreprocessDsConfig.model_validate(self._config_dict)

    def _load_data(self) -> None:
        self._df = self._injected_data.x.copy()
        self._df["target"] = self._injected_data.y.copy().apply(lambda x: int(x))

    def _handle_missing_values(self) -> None:
        method = self._config.missing_data_method
        self._log_param("preprocess - missing handling method", method)
        if not method:
            return

        numeric_cols = self._df.select_dtypes(include=[np.number]).columns
        categorical_cols = self._df.select_dtypes(
            include=["object", "category"]
        ).columns

        if method == "drop":
            self._df = self._df.dropna()
        elif method == "mean":
            imputer = SimpleImputer(strategy="mean")
            self._df[numeric_cols] = imputer.fit_transform(self._df[numeric_cols])
        elif method == "median":
            imputer = SimpleImputer(strategy="median")
            self._df[numeric_cols] = imputer.fit_transform(self._df[numeric_cols])
        elif method == "mode":
            for col in categorical_cols:
                self._df[col] = self._df[col].fillna(
                    self._df[col].mode()[0]
                    if not self._df[col].mode().empty
                    else "Unknown"
                )
            imputer = SimpleImputer(strategy="most_frequent")
            self._df[numeric_cols] = imputer.fit_transform(self._df[numeric_cols])
        elif method == "knn":
            imputer = KNNImputer(n_neighbors=5)
            self._df[numeric_cols] = imputer.fit_transform(self._df[numeric_cols])

    def _detect_outliers_iqr(self, df: pd.DataFrame, column: str) -> pd.Series:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)

    def _handle_outliers(self) -> None:
        method = self._config.outlier_method
        self._log_param("preprocess - outliers handling method", method)
        if not method:
            return

        numeric_cols = self._df.select_dtypes(include=[np.number]).columns
        if method == "remove":
            for col in numeric_cols:
                outliers = self._detect_outliers_iqr(self._df, col)
                self._df = self._df[~outliers]
        elif method == "cap":
            for col in numeric_cols:
                Q1 = self._df[col].quantile(0.25)
                Q3 = self._df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self._df[col] = self._df[col].clip(lower_bound, upper_bound)
        elif method == "transform":
            for col in numeric_cols:
                if (self._df[col] > 0).all():
                    self._df[col] = np.log1p(self._df[col])

    def _encode_categorical(self) -> None:
        self._log_param("preprocess - encoding method", "one hot encoding")
        categorical_cols = self._df.select_dtypes(
            include=["object", "category"]
        ).columns

        if len(categorical_cols.to_list()) > 0:
            for col in categorical_cols:
                if self._df[col].nunique() <= 10:
                    self._df = pd.get_dummies(
                        self._df, columns=[col], drop_first=True, prefix=col
                    )
                else:
                    self._df[col] = pd.factorize(self._df[col])[0]

    def _normalize_data(self) -> None:
        method = self._config.normalization_method
        self._log_param("preprocess - normalize handling method", method)
        if not self._config.normalization_method:
            return

        numeric_cols = self._df.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop("target")
        if method == "minmax":
            scaler = MinMaxScaler()
            self._df[numeric_cols] = scaler.fit_transform(self._df[numeric_cols])
        elif method == "standard":
            scaler = StandardScaler()
            self._df[numeric_cols] = scaler.fit_transform(self._df[numeric_cols])
        elif method == "robust":
            scaler = RobustScaler()
            self._df[numeric_cols] = scaler.fit_transform(self._df[numeric_cols])

    def _analyze_class_imbalance(self, y: pd.Series) -> dict:
        class_counts = y.value_counts()
        total_samples = len(y)

        imbalance_stats = {
            "class_distribution": class_counts.to_dict(),
            "total_samples": total_samples,
            "class_ratios": (class_counts / total_samples).to_dict(),
            "imbalance_ratio": (
                class_counts.max() / class_counts.min()
                if len(class_counts) > 1
                else 1.0
            ),
            "is_imbalanced": False,
        }

        if len(class_counts) > 1:
            minority_ratio = class_counts.min() / total_samples
            imbalance_ratio = class_counts.max() / class_counts.min()

            if imbalance_ratio > 1.5 or minority_ratio < 0.2:
                imbalance_stats["is_imbalanced"] = True

        return imbalance_stats

    def _handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        method = self._config.imbalance_method
        self._log_param("preprocess - imbalance handling method", method)
        if not self._config.imbalance_method:
            return X, y

        sampling_strategy = "auto"
        if method == "oversample":
            sampler = RandomOverSampler(
                sampling_strategy=sampling_strategy, random_state=self._seed
            )
            X_resampled, y_resampled = sampler.fit_resample(X, y)

        elif method == "undersample":
            sampler = RandomUnderSampler(
                sampling_strategy=sampling_strategy, random_state=self._seed
            )
            X_resampled, y_resampled = sampler.fit_resample(X, y)

        elif method == "smote":
            try:
                smote = SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=self._seed,
                    k_neighbors=min(5, len(y.unique()) - 1),
                )
                X_resampled, y_resampled = smote.fit_resample(X, y)
            except ValueError as e:
                self._logger.warning(
                    f"SMOTE failed: {e}. Using RandomOverSampler instead."
                )
                sampler = RandomOverSampler(random_state=self._seed)
                X_resampled, y_resampled = sampler.fit_resample(X, y)

        elif method == "smote_tomek":
            try:
                smote_tomek = SMOTETomek(
                    sampling_strategy=sampling_strategy,
                    random_state=self._seed,
                    smote=SMOTE(k_neighbors=min(5, len(y.unique()) - 1)),
                )
                X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
            except Exception as e:
                self._logger.warning(
                    f"SMOTETomek failed: {e}. Using RandomOverSampler instead."
                )
                sampler = RandomOverSampler(random_state=self._seed)
                X_resampled, y_resampled = sampler.fit_resample(X, y)

        elif method == "adasyn":
            try:
                from imblearn.over_sampling import ADASYN

                adasyn = ADASYN(
                    sampling_strategy=sampling_strategy,
                    random_state=self._seed,
                    n_neighbors=min(5, len(y.unique()) - 1),
                )
                X_resampled, y_resampled = adasyn.fit_resample(X, y)
            except Exception as e:
                self._logger.warning(
                    f"ADASYN failed: {e}. Using RandomOverSampler instead."
                )
                sampler = RandomOverSampler(random_state=self._seed)
                X_resampled, y_resampled = sampler.fit_resample(X, y)

        else:
            return X, y

        if isinstance(X_resampled, np.ndarray):
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        if isinstance(y_resampled, np.ndarray):
            y_resampled = pd.Series(y_resampled, name=y.name)

        return X_resampled, y_resampled

    def _split_data(self) -> tuple:
        X = self._df.drop(columns=["target"])
        y = self._df["target"].apply(lambda x: int(x))

        test_size = getattr(self._config, "test_size", 0.2)
        self._log_param("preprocess - split ratio", test_size)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self._seed, stratify=y, shuffle=True
        )

        return X_train, X_test, y_train, y_test

    @property
    def result(self) -> PreprocessResult:
        return self._result

    def run(self) -> None:
        self._handle_missing_values()
        self._handle_outliers()
        self._encode_categorical()
        self._normalize_data()

        X_train, X_test, y_train, y_test = self._split_data()
        X_train, y_train = self._handle_class_imbalance(X_train, y_train)

        train_result = LoadDsResult(
            x=X_train, y=y_train if y_train is not None else pd.Series(dtype="float64")
        )

        test_result = LoadDsResult(
            x=X_test, y=y_test if y_test is not None else pd.Series(dtype="float64")
        )

        self._result = PreprocessResult(train=train_result, test=test_result)

        if self._config.export_ds:
            self._export_datasets(X_train, X_test, y_train, y_test)

        if self._config.show:
            self._show_results()

    def _export_datasets(self, X_train, X_test, y_train, y_test) -> None:
        if y_train is not None:
            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)

        else:
            train_df = X_train
            test_df = X_test

        train_df.to_csv(self._output_path / "train.csv", index=False)
        test_df.to_csv(self._output_path / "test.csv", index=False)

        self._logger.info(f"Datasets exported to {self._output_path}/")

    def _show_results(self) -> None:
        """Display preprocessing results"""
        self._loger.info("=" * 50)
        self._loger.info("PREPROCESSING RESULTS")
        self._loger.info("=" * 50)
        imbalance_stats = self._analyze_class_imbalance(self._df["target"])
        self._loger.info("\nClass Imbalance Analysis:")
        self._loger.info(f"Class Distribution: {imbalance_stats['class_distribution']}")
        self._loger.info(f"Imbalance Ratio: {imbalance_stats['imbalance_ratio']:.2f}")
        self._loger.info(f"Is Imbalanced: {imbalance_stats['is_imbalanced']}")

        self._loger.info(f"\nOriginal data shape: {self._df.shape}")
        self._loger.info(f"Processed train shape: {self.result.train.x.shape}")
        self._loger.info(f"Processed test shape: {self.result.test.x.shape}")

        if self.result.train.y is not None and len(self.result.train.y) > 0:
            self._loger.info(f"\nTrain target distribution:")
            self._loger.info(self.result.train.y.value_counts())

        self._loger.info(f"\nMissing values after processing:")
        self._loger.info(f"Train: {self.result.train.x.isnull().sum().sum()}")
        self._loger.info(f"Test: {self.result.test.x.isnull().sum().sum()}")

        self._loger.info(f"\nData types after processing:")
        self._loger.info(self.result.train.x.dtypes.value_counts())
