# Datasets

This is a list of datasets stored in files.

## Challenger O-rings Dataset

The Challenger O-rings dataset counts the number of O-rings that failed in a
series of shuttle launches associated with the space shuttle Challenger mission.
It contains data for 23 launches counting the number of O-ring failures and the
temperature at the time of launch. The goal of the dataset is traditionally to
build a logistic regression model predicting the failure probability at the
temperature on the actual launch day, which was 36 °F.

- File: `challenger_orings.csv`
- Code: `osos.load_challenger(as_data_frame: bool)`

## Ames Housing Dataset

The Ames housing dataset is a tabular dataset consisting of 2930 observations
and 82 features which may be used to predict the selling price of a residential
property in Ames, Iowa, from 2006 to 2010. The target variable for regression
modeling is the `SalePrice`. Due to the number of features in the data, several
fields contain missing data that was either unobserved or undefined for the
particular parcel of property in question. Therefore a precursor to modeling the
data is to impute (fill in, complete) some of these missing values.

The data were originally obtained from the Ames, Iowa Assessor's Office, and are
commonly redistributed via
[Kaggle](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset).[^1]
More complete descriptions of the data are available from the [American
Statistical Association
(ASA)](https://jse.amstat.org/v19n3/decock/DataDocumentation.txt).[^2]

- File `ames-housing.csv`

## References

[^1]: Kaggle. Ames Housing Dataset.
    https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset

[^2]: De Cock, D. Ames Housing Dataset Description and references therein.
    https://jse.amstat.org/v19n3/decock/DataDocumentation.txt
