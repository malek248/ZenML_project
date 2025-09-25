import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
from typing import List, Dict, Tuple
import pandas as pd
from zenml import step
from typing_extensions import Annotated


@step
def find_best_weight(
    X_train: pd.DataFrame, y_train: pd.DataFrame, weights: List[int], cv: int
) -> Annotated[int, "best weight"]:
    """Cross‑validate over sample weights, pick best by positive‑class recall."""
    best_recall, best_w = -1, weights[0]
    for w in weights:
        model = XGBClassifier(
            use_label_encoder=False, eval_metric="auc", scale_pos_weight=w
        )
        preds = cross_val_predict(
            model,
            X_train,
            y_train,
            cv=cv,
            method="predict",
        )
        tn, fp, fn, tp = confusion_matrix(y_train, preds).ravel()
        recall_pos = tp / (tp + fn) if (tp + fn) else 0
        if recall_pos > best_recall:
            best_recall, best_w = recall_pos, w
    return best_w


@step
def train_initial_model(
    X_train: pd.DataFrame, y_train: pd.DataFrame, best_weight: int, xgb_base: Dict
) -> Annotated[XGBClassifier, "model"]:
    """Train your final XGBClassifier on the whole training set."""
    params = xgb_base.copy()
    params["scale_pos_weight"] = best_weight
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


@step
def select_features_transformer(
    model: XGBClassifier, feature_threshold: float
) -> SelectFromModel:
    """Wrap your fitted XGB into a SelectFromModel for feature‑selection."""
    return SelectFromModel(model, threshold=feature_threshold, prefit=True)


@step
def transform_train(selector: SelectFromModel, X_train: pd.DataFrame) -> np.ndarray:
    """Apply your selector to the training set."""
    return selector.transform(X_train)


@step
def transform_test(
    selector: SelectFromModel, X_test: pd.DataFrame
) -> pd.DataFrame:  # Changed return type hint
    """Apply your selector to the held-out test set."""
    transformed_data = selector.transform(X_test)
    # Convert NumPy array back to DataFrame to match ParquetDataset expectation
    return pd.DataFrame(transformed_data)


@step
def train_final_model(
    X_train_sel: np.ndarray, y_train: pd.DataFrame, best_weight: int, xgb_base: Dict
) -> Annotated[XGBClassifier, "final model"]:
    """Train your final XGBClassifier on the _selected_ training set."""
    params = xgb_base.copy()
    params["scale_pos_weight"] = best_weight
    model = XGBClassifier(**params)
    model.fit(X_train_sel, y_train)
    return model


@step
def predict_model(
    model: XGBClassifier, X_test_sel: pd.DataFrame
) -> Tuple[
    Annotated[list, "y_pred"], Annotated[list, "y_proba"]
]:  # Changed y_proba return type hint to List
    """Produce final y_pred and y_proba on the _selected_ test set."""
    y_proba = model.predict_proba(X_test_sel)[:, 1]
    y_pred = model.predict(X_test_sel)
    return y_pred.tolist(), y_proba.tolist()  # Convert both y_pred and y_proba to lists
