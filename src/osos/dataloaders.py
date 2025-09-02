"""dataloaders

Wrapper for opening datasets from `sklearn.datasets` [^1], with a separate loader for
the Challenger O-rings dataset. The `sklearn.datasets` module provides about 30
different mechanisms for loading, fetching, or generating data. This module exposes a
very small subset of those through a single interface. A description of the Challenger
data is appended after examples.

The intended interface is as follows:

```python
Task(Enum)
  CLASSIFICATION = 0
  REGRESSION = 1
  MULTIVARIATE_REGRESSION = 2

Datasets(Enum)
  BREAST_CANCER = (0, Task.CLASSIFICATION)
  CHALLENGER = (1, Task.CLASSIFICATION)
  DIABETES = (2, Task.REGRESSION)
  EXERCISE = (3, Task.MULTIVARIATE_REGRESSION)
  IRIS = (4, Task.CLASSIFICATION)
  WINE = (5, Task.CLASSIFICATION)

Dataset
  x: np.ndarray
  y: np.ndarray

DatasetInfo:
  description: str
  feature_names: list[str]
  num_samples: int
  num_features: int
  task_type: str
  notes: str | None = None
  reference: str | None = None

get_dataset_info(dataset: Datasets) -> DatasetInfo:
  '''Browse supported datasets by task to determine what data you want to analyze.'''


load_dataset(dataset: Datasets) -> tuple[Dataset, DatasetInfo]:
  '''Load one of the datasets provided in Datasets.'''

load_challenger(as_dataframe: bool = False) -> DataFrame | Dataset:
  '''Loads the Challenger dataset.'''
```

Examples
--------
# Querying a dataset
>>> ds_info = get_dataset_info(Datasets.BREAST_CANCER)
>>> print(ds_info)  # with formatting for readability here
#
# DatasetInfo(
#     description='Breast cancer tumors engineered from digitized images of fine needle aspirates. Features summarize traits of cell nuclei in the images',
#     feature_names=['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave points', 'symmetry', 'fractal dimension'],
#     num_samples=569,
#     num_features=30,
#     task_type='CLASSIFICATION',
#     notes='
#     1. Class distribution is {benign:357, malignant:212}
#     2. The dataset says print contains 30 features total, many of which are summary
#        statistics taken from imaging that is not included in the dataset. See documentation
#        from the scikit-learn authors for more details on the dataset.',
#    reference='https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset'
# )

# Loading a dataset
>>> ds, ds_info = load_dataset(Datasets.BREAST_CANCER)
>>> print(ds.X.shape)  # Shape(569, 30)
>>> print(ds.y.shape)  # Shape(569, )


Challenger Dataset:

    The orings dataset documents previous launches from the NASA Challenger mission,
    counting the number of O-rings (out of six) which failed during a mission. The
    Challenger mission ended in disaster on January 28, 1986: 73 seconds after the
    shuttle launched from Cape Canaveral, Florida, it disintegrated in the upper
    stratosphere of the Earth, killing all seven crew members on board the flight.

    Following the disaster, the Rogers Commission was formed to investigate the failure.
    The temperature in Fahrenheit for each mission was recorded with the number of
    orings that failed. The leading hypothesis for the cause of failure is that the
    orings became brittle in cold air, a point which Richard Feynman demonstrated on the
    floor of the US House of Representatives by comparing a room temperature O-ring with
    another O-ring that had been submerged in cold water.

    The Challenger mission included 23 flights prior to this disaster in which the
    temperature (in Fahrenheit) and the number of failed O-rings were recorded. These
    data are provided here sorted by temperature, ranging from 53--81 degrees F together
    with the number of O-rings that failed on that particular launch. Compared with the
    historical data, the launch temperature for the Jan 28, 1986, flight was 36 degrees.

    References:
        [1] https://openintro.org/data/index.php?data=orings
        [2] https://archive.ics.uci.edu/


References
----------

[^1] https://scikit-learn.org/stable/datasets.html#datasets
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from pandas import DataFrame
from sklearn import datasets


class Task(Enum):
    CLASSIFICATION = 0
    REGRESSION = 1
    MULTIVARIATE_REGRESSION = 2


class Datasets(Enum):
    """Loadable datasets from `sklearn.datasets`'s `load_*` methods

    This interface limits to loading datasets stored in `sklearn`. It does not expose
    any of the `fetch_*` methods because they require downloading compressed data
    formats from remote sources. For the workshop this is considered unideal.

    Use `get_dataset_info(dataset)` for detailed metadata about the supported datasets.

    The enum tabulates two pieces of information per dataset: an index (arbitrary) and
    the task type (C = classification, R = regression, M = multivariate regression).

    | Dataset        | Index | Type |
    | -------------- | ----- | ---- |
    | BREAST_CANCER  | 0     | C    |
    | CHALLENGER     | 1     | C    |
    | DIABETES       | 1     | R    |
    | EXERCISE       | 2     | M    |
    | IRIS           | 3     | C    |
    | WINE           | 4     | C    |
    """

    BREAST_CANCER = (0, Task.CLASSIFICATION)
    CHALLENGER = (1, Task.CLASSIFICATION)
    DIABETES = (1, Task.REGRESSION)
    EXERCISE = (2, Task.MULTIVARIATE_REGRESSION)
    IRIS = (3, Task.CLASSIFICATION)
    WINE = (4, Task.CLASSIFICATION)

    @property
    def task(self) -> Task:
        return self.value[1]

    @classmethod
    def by_task(cls, task: Task) -> list[Datasets]:
        return [ds for ds in cls if ds.task == task]


@dataclass
class Dataset:
    """Dataset container type for named access to data.

    The fields denote data for any model of the type y = f(x).

    Fields:
        x: Predictor variables to model the response y.
        y: The targets/labels to be modeled.
    """

    x: np.ndarray
    y: np.ndarray


@dataclass
class DatasetInfo:
    description: str
    feature_names: list[str]
    target_name: dict[str, str | int]
    num_samples: int
    num_features: int
    task_type: str
    notes: str | None = None
    reference: str | None = None


DATASET_LOADERS: dict[Datasets, Callable] = {
    Datasets.BREAST_CANCER: datasets.load_breast_cancer,
    Datasets.CHALLENGER: lambda: load_challenger(as_dataframe=False),
    Datasets.DIABETES: datasets.load_diabetes,
    Datasets.EXERCISE: datasets.load_linnerud,
    Datasets.IRIS: datasets.load_iris,
    Datasets.WINE: datasets.load_wine,
}

DATASET_INFO: dict[Datasets, DatasetInfo] = {
    Datasets.BREAST_CANCER: DatasetInfo(
        description="Breast cancer tumors engineered from digitized images of fine needle aspirates. Features summarize traits of cell nuclei in the images",
        feature_names=[
            "radius",
            "texture",
            "perimeter",
            "area",
            "smoothness",
            "compactness",
            "concavity",
            "concave points",
            "symmetry",
            "fractal dimension",
        ],
        target_name={"malignant": 0, "benign": 1},
        num_samples=569,
        num_features=30,
        task_type=Task.CLASSIFICATION.name,
        notes=(
            "1. Class distribution is {benign:357, malignant:212}\n2. The dataset says "
            "print contains 30 features total, many of which are summary statistics "
            "taken from imaging that is not included in the dataset. See documentation "
            "from the scikit-learn authors for more details on the dataset."
        ),
        reference="https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset",
    ),
    Datasets.CHALLENGER: DatasetInfo(
        description="O-ring and launch data from the NASA Challenger shuttle mission. Data provide a glimpse into the failure conditions of previous launches leading up to the 1986 disaster.",
        feature_names=["temperature"],
        target_name={"successful-launch": 1, "failure": 0},
        num_samples=23,
        num_features=1,
        task_type=Task.CLASSIFICATION.name,
        notes="1. Temperatures are provided in Fahrenheit. A dataset with more information is available by calling `load_challenger(as_data_frame=True)` to access the same data.\n2. The temperature for the shuttle failure is not included in the data, but was was recorded as 36 F for the Jan 28, 1986, launch.",
        reference="https://www.openintro.org/data/index.php?data=orings",
    ),
    Datasets.IRIS: DatasetInfo(
        description="The Iris database introduced by RA Fisher. The original work developed linear discriminant analysis",
        feature_names=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
        target_name={
            "species-setosa": 0,
            "species-versicolor": 1,
            "species-virginica": 2,
        },
        num_samples=150,
        num_features=4,
        task_type=Task.CLASSIFICATION.name,
    ),
    Datasets.WINE: DatasetInfo(
        description="Chemical analyses of wines grown in the same region by three vineyards",
        feature_names=[
            "alcohol",
            "malic acid",
            "ash",
            "alcalinity of ash",
            "magnesium",
            "total phenols",
            "flavanoids",
            "nonflavanoid phenols",
            "proanthocyanins",
            "color intensity",
            "hue",
            "od280/od315 of diluted wines",
            "proline",
        ],
        target_name={"class_0": 0, "class_1": 1, "class_2": 2},
        num_samples=178,
        num_features=13,
        task_type=Task.CLASSIFICATION.name,
        notes="1. Class distributions are {0:59, 1:71, 2:48}.",
        reference="https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
    ),
    Datasets.DIABETES: DatasetInfo(
        description="Predicing progression of diabetes one year after baseline",
        feature_names=[
            "age (years)",
            "sex",
            "body mass index (bmi)",
            "avg blood pressure (bp)",
            "total serum, cholesterol (s1 tc)",
            "low-density lipoproteins (s2 ldl)",
            "high density lipoproteins (s3 hdl)",
            "total cholesterol / hdl (s4 tch)",
            "possibly log serum triglycerides level (s5 ltg)",
            "blood sugar level (s6 glu)",
        ],
        target_name={"disease_progression": "one_year_after_baseline"},
        num_features=10,
        num_samples=442,
        task_type=Task.REGRESSION.name,
    ),
    Datasets.EXERCISE: DatasetInfo(
        description=(
            "Multi-output regression dataset using three exercise (response/target) "
            "variables and three physiological (feature) variables"
        ),
        feature_names=[
            "weight",
            "waist",
            "pulse",
        ],
        target_name={"exercise": "chinups,situps,jumps"},
        num_samples=20,
        num_features=3,
        task_type=Task.MULTIVARIATE_REGRESSION.name,
        notes=(
            "1. This is a multitarget regression problem with three response variables. "
            "Predictions should use three feature variables which are expected to be "
            "correlated (perhaps not linearly) with the targets."
        ),
    ),
}


def get_dataset_info(dataset: Datasets) -> DatasetInfo:
    """Browse supported datasets by task to determine what data you want to analyze.

    Datasets:
        BREAST_CANCER (classification)
        DIABETES      (regression)
        EXERCISE      (multivariate regression)
        IRIS          (classification)
        WINE          (classification)

    Examples:
    ---------
        >>> ds_info = get_dataset_info(Datasets.BREAST_CANCER)
        >>> print(ds_info)
        #
        # Returns
        #
        # DatasetInfo(
        #     callback='sklearn.datasets.load_breast_cancer',
        #     description='Breast cancer tumors engineered from digitized images of fine needle aspirates. Features summarize traits of cell nuclei in the images',
        #     feature_names=['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave points', 'symmetry', 'fractal dimension'],
        #     num_samples=569,
        #     num_features=30,
        #     task_type='CLASSIFICATION',
        #     notes='
        #     1. Class distribution is {benign:357, malignant:212}
        #     2. The dataset says print contains 30 features total, many of which are summary
        #        statistics taken from imaging that is not included in the dataset. See documentation
        #        from the scikit-learn authors for more details on the dataset.',
        #    reference='https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset'
        # )
    """
    ds_info = DATASET_INFO[dataset]
    return ds_info


def load_dataset(dataset: Datasets) -> tuple[Dataset, DatasetInfo]:
    """Load one of the datasets provided in Datasets.

    Datasets:
        BREAST_CANCER (classification)
        DIABETES      (regression)
        EXERCISE      (multivariate regression)
        IRIS          (classification)
        WINE          (classification)

    Examples:
        >>> ds, ds_info = load_dataset(Datasets.BREAST_CANCER)
        >>> print(ds.x.shape)       # Shape(539, 30)
        >>> print(ds.y.shape)       # Shape(539,)
    """
    data = DATASET_LOADERS[dataset]()
    if dataset.name == "CHALLENGER":
        ds = data
    else:
        ds = Dataset(data.data, data.target)
    ds_info = DATASET_INFO[dataset]
    return ds, ds_info


def load_challenger(as_dataframe: bool = False) -> DataFrame | Dataset:
    """Loads the O-rings dataset as a pandas.DataFrame or Dataset.

    The orings dataset documents previous launches from the NASA Challenger mission,
    counting the number of O-rings (out of six) which failed during a mission. The
    Challenger mission ended in disaster on January 28, 1986: 73 seconds after the
    shuttle launched from Cape Canaveral, Florida, it disintegrated in the upper
    stratosphere of the Earth, killing all seven crew members on board the flight.

    Following the disaster, the Rogers Commission was formed to investigate the failure.
    The temperature in Fahrenheit for each mission was recorded with the number of
    orings that failed. The leading hypothesis for the cause of failure is that the
    orings became brittle in cold air, a point which Richard Feynman demonstrated on the
    floor of the US House of Representatives by comparing a room temperature O-ring with
    another O-ring that had been submerged in cold water.

    The Challenger mission included 23 flights prior to this disaster in which the
    temperature (in Fahrenheit) and the number of failed O-rings were recorded. These
    data are provided here sorted by temperature, ranging from 53--81 degrees F together
    with the number of O-rings that failed on that particular launch. Compared with the
    historical data, the launch temperature for the Jan 28, 1986, flight was 36 degrees.

    Args:
        as_dataframe: Load a DataFrame (True) or Dataset (False).

    Returns:
        Either a pandas.DataFrame (as_dataframe=True) or a Dataset (as_dataframe=False).
        - DataFrame: If a DataFrame is requested then this function provides the csv
          table (orings.csv) from [1] based on an archive of the University of
          California Irvine (UCI) machine learning datasets repository [2]; note this
          table requires processing to use in any modeling. The table fields are
          * mission number: index, arbitrary but sorted by temperature
          * temperature: degrees Fahrenheit, corresponding to the temperature during
            launch
          * damaged: the number of orings damanged (out of 6)
          * undamaged: the number of orings undamaged (out of 6)
        - Dataset: If a Dataset is requested then it is returned in a tidy format to be
          compatible with the `sklearn` interface primarily in mind. The Dataset is
          designed to be a plug-and-play dataset.

    References:
        [1] https://openintro.org/data/index.php?data=orings
        [2] https://archive.ics.uci.edu/
    """
    mission = np.arange(1, 24, dtype=np.uint8)
    temp = np.array([53, 57, 58, 63, 66, 67, 67, 67, 68, 69, 70, 70, 70, 70, 72, 73, 75, 75, 76, 76, 78, 79, 81], dtype=np.uint8)  # fmt: skip
    damaged = np.array([5, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.uint8)  # fmt: skip
    undamaged = np.array([1, 5, 5, 5, 6, 6, 6, 6, 6, 6, 5, 6, 5, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6], dtype=np.uint8)  # fmt: skip

    if as_dataframe:
        return DataFrame(
            dict(mission=mission, temp=temp, damaged=damaged, undamaged=undamaged),
            index=mission,
            dtype=np.uint8,
        ).set_index("mission")
    else:
        return Dataset(temp[:, None].copy(), (damaged != 0).astype(np.uint8))
