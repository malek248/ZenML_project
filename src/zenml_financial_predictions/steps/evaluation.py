import numpy as np
from zenml import step
import pandas as pd
from typing_extensions import Annotated


@step
def evaluate_model(
    y_test: pd.DataFrame,
    y_pred: list,
    y_proba: list,
) -> Annotated[dict[str, float], "metrics"]:
    """Compute and return a small metrics dict."""
    from sklearn.metrics import recall_score, precision_score, roc_auc_score

    return {
        "recall_pos": recall_score(y_test, y_pred, pos_label=1),
        "precision_pos": precision_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
