## ZenML
from zenml import step
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated


@step
def load_data() -> pd.DataFrame:
    """Load the raw data from CSV."""
    df = pd.read_csv("./data/bank.csv", sep=";", quotechar='"')
    return df


@step
def clean_data(bank: pd.DataFrame) -> Annotated[pd.DataFrame, "clean data"]:
    print("Columns I actually got:", bank.columns.tolist())
    bank = bank.dropna()
    bank["day"] = bank["day"].astype("object")

    return bank


@step
def split_raw_data(
    data_raw_all: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[Annotated[pd.DataFrame, "data train"], Annotated[pd.DataFrame, "data test"]]:
    data_train, data_test = train_test_split(
        data_raw_all, test_size=test_size, random_state=random_state
    )
    return data_train, data_test
