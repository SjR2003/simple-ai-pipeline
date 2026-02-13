from pydantic import BaseModel
from typing import Optional


class PreprocessDsConfig(BaseModel):
    task_name: str
    export_ds: bool
    show: bool

    test_size: float = 0.2
    label_map: Optional[dict] = None
    label_map_strict: bool = False
    missing_data_method: Optional[str] = None  # 'drop', 'mean', 'median', 'mode', 'knn'
    outlier_method: Optional[str] = None  # 'remove', 'cap', 'transform'
    imbalance_method: Optional[str] = (
        None  # 'oversample', 'undersample', 'smote', 'smote_tomek', 'adasyn'
    )
    normalization_method: Optional[str] = None  # 'minmax', 'standard', 'robust'
    model_config = {"arbitrary_types_allowed": True}
