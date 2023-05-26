from typing import NamedTuple


class _REGISTRY_KEYS_VT(NamedTuple):
    X_KEY: str = "X"  # total
    N_KEY: str = "N"  # new
    U_KEY: str = "U"  # unspliced (for splicing variant of model)
    KNN_KEY: str = "knn"  # indices of kNNs
    TS_KEY: str = "ts"
    BATCH_KEY: str = "batch"
    LABELS_KEY: str = "labels"
    CAT_COVS_KEY: str = "extra_categorical_covs"
    CONT_COVS_KEY: str = "extra_continuous_covs"
    INDICES_KEY: str = "ind_x"
    SIZE_FACTOR_KEY: str = "size_factor"


class _REGISTRY_KEYS_SDE(NamedTuple):
    X_KEY: str = "x"  # total
    TIME_KEY: str = "t"  # time metric used for SDE modelling
    INDEX_KEY: str = "index"
    BATCH_KEY: str = "batch"
    CAT_COVS_KEY: str = "extra_categorical_covs"
    CONT_COVS_KEY: str = "extra_continuous_covs"


REGISTRY_KEYS_VT = _REGISTRY_KEYS_VT()
REGISTRY_KEYS_SDE = _REGISTRY_KEYS_SDE()
