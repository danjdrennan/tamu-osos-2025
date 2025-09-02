from importlib.metadata import version, PackageNotFoundError

# Package versioning
try:
    __version__ = version("osos")
except PackageNotFoundError:
    __version__ = "development"

from ._timer import timer

from .dataloaders import (
    Datasets,
    Dataset,
    DatasetInfo,
    get_dataset_info,
    load_challenger,
    load_dataset,
)

from .config_hashing import (
    LoggingLevel,
    hash_config_and_data,
)
