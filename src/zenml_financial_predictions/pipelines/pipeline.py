import pathlib
import sys
_project_src_dir = pathlib.Path(__file__).resolve().parent.parent.parent

if str(_project_src_dir) not in sys.path:
    sys.path.insert(0, str(_project_src_dir))

from zenml_financial_predictions.steps.data_processing import (
    load_data,
    clean_data,
    split_raw_data,
)
from zenml_financial_predictions.steps.feature_engineering import (
    split_attributes,
    transform_features,
    transform_target,
)
from zenml_financial_predictions.steps.model_training import (
    find_best_weight,
    train_initial_model,
    select_features_transformer,
    transform_train,
    transform_test,
    train_final_model,
    predict_model,
)
from zenml_financial_predictions.steps.evaluation import evaluate_model
from zenml import pipeline


@pipeline
def financial_pipeline():
    """Pipeline for financial predictions."""
    # 1. Load & clean
    df_raw = load_data()
    df_clean = clean_data(df_raw)

    # 2. Split
    train_df, test_df = split_raw_data(
        df_clean,
        test_size=0.2,
        random_state=42,
    )

    # 3. Feature engineering
    attrs_map = split_attributes(train_df)
    X_train = transform_features(train_df, attrs_map)
    y_train = transform_target(train_df, attrs_map)
    X_test = transform_features(test_df, attrs_map)
    y_test = transform_target(test_df, attrs_map)

    # 4. Model training
    best_w = find_best_weight(
        X_train, y_train, weights=[1, 5, 20, 50, 100, 1000, 10000], cv=5
    )
    init_model = train_initial_model(
        X_train,
        y_train,
        best_w,
        xgb_base={
            "scale_pos_weight": 1,
            "use_label_encoder": False,
            "eval_metric": "auc",
        },
    )
    selector = select_features_transformer(init_model, feature_threshold=0.1)
    X_train_sel = transform_train(selector, X_train)
    final_model = train_final_model(
        X_train_sel,
        y_train,
        best_w,
        xgb_base={
            "scale_pos_weight": 1,
            "use_label_encoder": False,
            "eval_metric": "auc",
        },
    )
    X_test_sel = transform_test(selector, X_test)
    y_pred, y_proba = predict_model(final_model, X_test_sel)

    # 5. Evaluation
    evaluate_model(y_test, y_pred, y_proba)


if __name__ == "__main__":
    financial_pipeline()
