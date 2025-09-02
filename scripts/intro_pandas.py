from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import osos

# This is just a constant defining a file path. It will make our lives easier if we
# define it now and then gather data along relative paths from the data.
ROOT = Path(".")
DATA = ROOT / "data"

# This is configuring the print options for pandas so that we can actually see results
pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns

# This module covers pandas, a library for supercharging tabular data analyses. We
# cannot possibly cover everything this library can do in a day, let alone a workshop.
# So we are going to get into some of the very basics of loading a dataset and exploring
# it. For this module we are going to augment the standard plotting imports to use
# `seaborn`, which is a handy tool for producing the specific plots we'd like to see in
# this dataset.

# For starters, let's load the iris dataset. This is a historical dataset first
# introduced by Sir Ronald A Fisher which is commonly used for introductory
# classification and visualization tasks. It's a very sterile dataset, making it a good
# starting point.

# read_csv from DATA / 'iris.csv' into a dataframe `df`
df = pd.read_csv(DATA / "iris.csv")

# The first step after loading any dataset is to inspect the contents. There are a
# number of things we'd like to know to get started, such as
#
# - how many rows and columns there are in the dataframe
# - what the contents of those columns are
# - whether there are any missing values
# - summary statistics about the data
# - any hierarchical structure we might expect to find
#
# We can see some of this by printing the 'head' and 'tail' of the file, which gives us
# a preview of the first or last few rows of the table.
print(df.head())
print(df.tail())

# We can gather a more complete picture by calling the following methods:
print(df.info())
print(f"\nmissing data?\n{df.isnull().any()}")
print(f"(rows,cols): {df.shape}")
print(df.describe())
print(df.columns)

# As promised, this is a very clean dataset that's easy to work with. Let's next look at
# a visualization. For context, this dataset was carefully curated to demonstrate
# various kinds of ANOVA designs and linear discriminant analysis.

# This is called a pairplot or pairs plot. What it shows is a panel of all of the
# variables in the dataframe. The diagonals are histograms/density plots of the marginal
# distributions for each variable. The off-diagonals are the interactions between each
# marginal variable. In this way, we are able to inspect every pairwise dependency in
# the dataset. Notice also that we have colored the variables by the different species,
# so we are actually seeing three subpopulations within this sample. And what that shows
# is that there are distinct differences in the relationships between the variables in
# this dataset.
if True:
    palette = sns.palettes.color_palette("pastel", n_colors=3)
    sns.pairplot(df, hue="species", palette=palette)
    plt.show()

# Alright, that is most of the exploration we wanted to do. Now let's look at some of
# the other important features of a pandas dataframe. For starters, this is a table that
# we may wish to query. Since we have a categorical variable ('species'), one of the
# things we may like to do is calculate summary statistics conditioned on that variable.


if True:
    subgroups = df.groupby("species").agg(
        sepal_length_mean=("sepal_length", "mean"),
        sepal_length_std=("sepal_length", "std"),
        sepal_width_mean=("sepal_width", "mean"),
        sepal_width_std=("sepal_width", "std"),
        petal_length_mean=("petal_length", "mean"),
        petal_length_std=("petal_length", "std"),
        petal_width_mean=("petal_width", "mean"),
        petal_width_std=("petal_width", "std"),
    )
    # print(subgroups.round(2))

    # If we only wanted to see the means, we could *query* the dataframe for those values.
    print(subgroups.filter(regex="mean").round(2))
    print(subgroups.filter(regex="std").round(2))

# If we wanted to filter the original data to examine a particular subpopulation, we
# could do that as well:
print(df[df.species == "Iris-setosa"])

# And we can combine queries to form even more complex views of our data. Just note that
# this is a boolean algebra, so we use multiplication for AND relationships.

# If we wanted to find all of the samples which are setosa with a sepal length greater
# than 5.5, how would we do that?
print(df[(df.species == "Iris-setosa") * (df.sepal_length > 5.5)])
print(df[(df.species == "Iris-setosa") & (df.sepal_length > 5.5)])

# But there's actually a slightly nicer querying syntax we can use either of these:

species_name = "Iris-setosa"
threshold = 5.5
print(df.query("species == @species_name and sepal_length > @threshold"))

print(df.query("species == 'Iris-setosa' and sepal_length > 5.5"))

# The other querying tools that are nice to know are `df.loc` and `df.iloc`. Using `loc`
# lets you query data by label whereas iloc lets you index by integer values.

# This lets you query data by the row index and column of the dataframe using names for
# the columns. If the row index were something like a series of dates then we could
# query by the dates directly rather than using an integer index.
print(df.loc[0, "species"])

# The iloc version of this is a lower level indexing operation which requires us to
# actually understand where things are located in our table. Here's the same query using
# iloc, noting that the species is the last column of the data.
print(df.iloc[0, -1])

# Something we have not done up to this point, but which is useful, is to augment the
# data with other features. But whenever you do this, note it is generally a good idea
# to clone the dataframe in the process. The default behavior in pandas is to return new
# clones of data rather than modifying things in place. That can be overridden using the
# `inplace=True` argument to any function. However, it is generally better to copy the
# dataframe to maintain provenance over whatever transformations we've been doing. The
# exception to this rule is the case when we are working with data that occupies a
# significant fraction of the computer memory we have available.

# Here is a simple injection of a new variable into the dataframe. This one is
# meaningless in this context, but it could be useful in some contexts.
df["sepal_length_centered"] = df.sepal_length - df.sepal_length.mean()


# Since we have subgroups, a better comparison is to center the variables by subgroup.
# This is one way of doing that:
def center_by_subgroup(data, variable, groupby_col):
    return data[variable] - data.groupby(groupby_col)[variable].transform("mean")


# df["sepal_length_centered_by_subgroup"] = center_by_subgroup(
#     df, "sepal_length", "species"
# )

df["sepal_length_centered_by_subgroup"] = df["sepal_length"] - df.groupby("species")[
    "sepal_length"
].transform("mean")

print(df.iloc[..., -3:])

sns.pairplot(df.iloc[:, -3:], hue="species")
plt.show()
