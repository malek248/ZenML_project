# Exploring the type of the different variables of the dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from zenml import step
from uuid import UUID

# Workaround for seaborn bug looking for Legend.legendHandles instead of legend_handles
from matplotlib.legend import Legend

Legend.legendHandles = property(lambda self: self.legend_handles)


@step
def get_dtypes(df: pd.DataFrame) -> pd.Series:
    return df.dtypes


# Positive and negative class distribution


@step
def plot_count(df: pd.DataFrame) -> plt:
    """
    Draw and return a count plot of the target variable.
    """
    sns.countplot(x="y", data=df, palette="husl")
    plt.title("Count Plot of 'y'")
    # Let MatplotlibWriter handle saving/closing
    return plt


# Exploring the numeric column


@step
def groupby_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("y").mean(numeric_only=True)


# Plotting the distribution of the features and their cross co-relation


@step
def plot_pairplot(df: pd.DataFrame, num_cols_with_y: list) -> plt:
    """
    Draw and return a pair plot with KDE on the diagonal.
    Legend is removed afterward to avoid the Legend.legendHandles bug.
    """
    # 1) Create the grid
    g = sns.pairplot(
        df[num_cols_with_y],
        hue="y",
        diag_kind="kde",  # KDE on the diagonal
        dropna=True,
        markers=[",", ","],
        palette=sns.color_palette(["red", "green"]),
        plot_kws={"s": 3},
    )

    # 2) Remove the automatically added legend (avoids Legend.legendHandles issues)
    if g._legend:
        g._legend.remove()

    # 3) Add KDE contours on the lower triangle
    g.map_lower(sns.kdeplot, cmap="Blues_d")

    # 4) Adjust the figure title
    g.fig.suptitle("Pair Plot", y=1.02)

    return plt


# The corelation between inputs


@step
def plot_heatmap(df: pd.DataFrame, num_cols: list) -> Figure:  # Updated type hint
    """
    Draw and return a heatmap of correlations among numeric features.
    """
    plt.figure()  # Ensure a new figure is created for this plot
    corr = df[num_cols].corr()
    sns.heatmap(
        corr,
        xticklabels=corr.columns.values,
        yticklabels=corr.columns.values,
        cmap=sns.light_palette("navy"),
    )
    plt.title("Correlation Heatmap")
    # Let MatplotlibWriter handle saving/closing
    return plt.gcf()  # Return the current figure object
