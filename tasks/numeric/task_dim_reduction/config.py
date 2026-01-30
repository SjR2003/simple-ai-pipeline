from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ReductionMethod(str, Enum):
    PCA = "pca"
    TSNE = "tsne"
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

    tsne_perplexity: float = Field(
        default=30.0, ge=5.0, le=50.0, description="Perplexity parameter for t-SNE"
    )
    tsne_n_iter: int = Field(
        default=1000, ge=250, le=5000, description="Number of iterations for t-SNE"
    )
    tsne_init: str = Field(
        default="random", description="Initialization method for t-SNE"
    )
    tsne_use_pca_init: bool = Field(
        default=True, description="Use PCA initialization for high-dim data"
    )

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
