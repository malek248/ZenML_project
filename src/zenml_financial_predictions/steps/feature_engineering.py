import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.utils import check_array
from scipy import sparse
import numpy as np
from zenml import step
import ast  # Add import for ast
from typing_extensions import Annotated

# Utility functions


def get_data_attrs_names(data: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row DataFrame whose columns hold lists of attribute names."""
    all_attribs = list(data.columns)
    target_attrib = ["y"]

    # Categorical (drop the target if present)
    data_cat = data.select_dtypes(include=["object"]).drop(
        columns=target_attrib, errors="ignore"
    )
    cat_attribs = list(data_cat.columns)

    # Numerical
    num_attribs = list(data.select_dtypes(include=[np.number]).columns)

    # All predictors
    X_attribs = [c for c in all_attribs if c not in target_attrib]

    # wrap each list in another list → one-row DataFrame
    return pd.DataFrame(
        {
            "y": [target_attrib],
            "X_cat": [cat_attribs],
            "X_num": [num_attribs],
            "X": [X_attribs],
        }
    )


# Transformer classes


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class DataFrameCatImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series(
            [
                (
                    X[c].value_counts().index[0]
                    if X[c].dtype == np.dtype("O")
                    else X[c].mean()
                )
                for c in X
            ],
            index=X.columns,
        )
        return self

    def transform(self, X):
        return X.fillna(self.fill)


class MyLabelEncoder(LabelEncoder):
    def fit_transform(self, X, y=None):
        return super(MyLabelEncoder, self).fit_transform(X)

    def fit(self, X, y=None):
        return super(MyLabelEncoder, self).fit(X)

    def transform(self, X):
        return super(MyLabelEncoder, self).transform(X)


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        encoding="onehot",
        categories="auto",
        dtype=np.float64,
        handle_unknown="error",
    ):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        if self.encoding not in ["onehot", "onehot-dense", "ordinal"]:
            raise ValueError("Invalid encoding type")
        if self.handle_unknown not in ["error", "ignore"]:
            raise ValueError("Invalid handle_unknown type")
        if self.encoding == "ordinal" and self.handle_unknown == "ignore":
            raise ValueError(
                "handle_unknown='ignore' not supported for ordinal encoding"
            )

        X = check_array(X, accept_sparse="csc", copy=True, dtype=object)
        n_samples, n_features = X.shape
        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == "auto":
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == "error":
                        diff = np.unique(Xi[~valid_mask])
                        raise ValueError(
                            f"Found unknown categories {diff} in column {i}"
                        )
                le.classes_ = np.array(np.sort(self.categories[i]))
        self.categories_ = [le.classes_ for le in self._label_encoders_]
        return self

    def transform(self, X):
        X = check_array(X, accept_sparse="csc", copy=True, dtype=object)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=int)
        X_mask = np.ones_like(X, dtype=bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])
            if not np.all(valid_mask):
                if self.handle_unknown == "error":
                    diff = np.unique(X[~valid_mask, i])
                    raise ValueError(f"Found unknown categories {diff} in column {i}")
                else:
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == "ordinal":
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32), n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix(
            (data, (row_indices, column_indices)),
            shape=(n_samples, indices[-1]),
            dtype=self.dtype,
        ).tocsr()
        if self.encoding == "onehot-dense":
            return out.toarray()
        else:
            return out


# Pipeline builders


def create_X_ml_pipeline(cat_attrs, num_attrs):
    num_pipeline = Pipeline(
        [
            ("selector", DataFrameSelector(num_attrs)),
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("selector", DataFrameSelector(cat_attrs)),
            ("cat_enc", CategoricalEncoder(encoding="onehot-dense")),
        ]
    )

    full_pipeline = FeatureUnion(
        transformer_list=[
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline),
        ]
    )

    return full_pipeline


def create_y_ml_pipeline(target_attr):
    target_pipeline = Pipeline(
        [
            ("selector", DataFrameSelector(target_attr)),
            ("label_enc", MyLabelEncoder()),
        ]
    )
    return target_pipeline


@step
def split_attributes(data_raw: pd.DataFrame) -> Annotated[pd.DataFrame, "attrs names"]:
    """
    Return a one‐row DataFrame whose columns hold the lists of names.
    """
    return get_data_attrs_names(data_raw)


@step
def transform_features(
    data_raw: pd.DataFrame, attrs_map: pd.DataFrame
) -> Annotated[pd.DataFrame, "transformed data features"]:  # Changed return type hint
    cat_attrs_val = attrs_map.loc[0, "X_cat"]
    num_attrs_val = attrs_map.loc[0, "X_num"]

    # Ensure cat_attrs and num_attrs are lists
    cat_attrs = (
        ast.literal_eval(cat_attrs_val)
        if isinstance(cat_attrs_val, str)
        else cat_attrs_val
    )
    num_attrs = (
        ast.literal_eval(num_attrs_val)
        if isinstance(num_attrs_val, str)
        else num_attrs_val
    )

    pipeline = create_X_ml_pipeline(cat_attrs=cat_attrs, num_attrs=num_attrs)
    transformed_data = pipeline.fit_transform(data_raw)
    # Convert NumPy array back to DataFrame to match ParquetDataset expectation
    # Note: Column names are lost in transformation, using default integer names.
    # For preserving names, the pipeline/node would need more complex handling.
    return pd.DataFrame(transformed_data)


@step
def transform_target(
    data_raw: pd.DataFrame, attrs_map: pd.DataFrame
) -> Annotated[pd.DataFrame, "transformed data target"]:  # Changed return type hint
    target_col_val = attrs_map.loc[0, "y"]

    # Ensure target_col is a list
    target_col = (
        ast.literal_eval(target_col_val)
        if isinstance(target_col_val, str)
        else target_col_val
    )

    pipeline = create_y_ml_pipeline(target_attr=target_col)
    transformed_data = pipeline.fit_transform(data_raw)
    # Convert NumPy array back to DataFrame to match ParquetDataset expectation
    # Using the original target column name for the DataFrame
    return pd.DataFrame(transformed_data, columns=target_col)
