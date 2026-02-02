from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ReductionMethod(str, Enum):
    PCA = "pca"
    UMAP = "umap"
    LDA = "lda"


class DimReductionConfig(BaseModel):
    task_name: str

    method: ReductionMethod = Field(
        default=ReductionMethod.PCA, description="Dimension reduction method"
    )
    n_components: int = Field(
        default=2, ge=2, le=100, description="Number of components to reduce to"
    )

    show: bool = Field(default=True, description="Show visualization plots")

    pca_svd_solver: str = Field(default="auto", description="SVD solver for PCA")

    umap_n_neighbors: int = Field(
        default=15, ge=2, le=100, description="Number of neighbors for UMAP"
    )
    umap_min_dist: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Minimum distance for UMAP"
    )
    umap_metric: str = Field(
        default="euclidean", description="Distance metric for UMAP"
    )
    umap_supervised: bool = Field(
        default=False, description="Use supervised UMAP when labels available"
    )

    lda_solver: str = Field(default="svd", description="Solver for LDA")
    lda_shrinkage: Optional[str] = Field(
        default=None, description="Shrinkage parameter for LDA"
    )
