from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec, ParamSchema, ParamSpec
from core.config import get_settings
from core.database import engine
import pandas as pd
import numpy as np
import os, time, mlflow
import warnings
warnings.filterwarnings("ignore")

# Config
settings = get_settings()
FEAST_REPO = "/repo"
MODEL_NAME = "churn_v1"
HYPERPARAMS = {
    "AS_OF": settings.AS_OF,
    "n_estimators": 300,
    "n_jobs": -1,
    "random_state": 42,
    "class_weight": "balanced",
    "max_features": "sqrt",
    "split_size": 0.25
}

def fetch_entity_df(engine, as_of):
    q = """
    SELECT user_id, as_of
    FROM subscriptions_profile_snapshots
    WHERE as_of = %(as_of)s
    """
    df = pd.read_sql(q, engine, params={"as_of": as_of})
    if df.empty:
        raise RuntimeError(f"No snapshot rows found at as_of={as_of}")
    df = df.rename(columns={"as_of": "event_timestamp"})
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    return df[["user_id", "event_timestamp"]]

def fetch_labels(engine, as_of: str) -> pd.DataFrame:
    query = "SELECT user_id, churn_label FROM labels"
    labels = pd.read_sql(query, engine)
    if labels.empty:
        raise RuntimeError("Labels table is empty.")
    labels["event_timestamp"] = pd.to_datetime(as_of)
    return labels[["user_id", "event_timestamp", "churn_label"]]

def build_training_set(store: FeatureStore, entities: pd.DataFrame, features: list[str]):
    histo_features = store.get_historical_features(
        entity_df=entities,
        features=features
    )
    return histo_features.to_df()


def prepare_dataset(dataframe, label_col="churn_label"):
    y = dataframe[label_col].astype(int).values
    X = dataframe.drop(columns=[label_col, "user_id", "event_timestamp"], errors="ignore")
    return X, y


if __name__ == "__main__":
    # mlflow setup
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT)

    entities = fetch_entity_df(engine=engine, as_of=settings.AS_OF)
    labels = fetch_labels(engine=engine, as_of=settings.AS_OF)
    features = [
        "subscription_profile_fv:months_active",
        "subscription_profile_fv:monthly_fee",
        "subscription_profile_fv:paperless_billing",
        "subscription_profile_fv:plan_stream_tv",
        "subscription_profile_fv:plan_stream_movies",
        "subscription_profile_fv:net_service",
        "usage_agg_30d_fv:watch_hours_30d",
        "usage_agg_30d_fv:avg_session_mins_7d",
        "usage_agg_30d_fv:unique_devices_30d",
        "usage_agg_30d_fv:skips_7d",
        "usage_agg_30d_fv:rebuffer_events_7d",
        "payments_agg_90d_fv:failed_payments_90d",
        "support_agg_90d_fv:support_tickets_90d",
        "support_agg_90d_fv:ticket_avg_resolution_hrs_90d",
    ]
    store = FeatureStore(repo_path=FEAST_REPO)
    features_dataframe = build_training_set(store=store, entities=entities, features=features)
    dataset = features_dataframe.merge(labels, on=["user_id", "event_timestamp"], how="inner")
    if dataset.empty:
        raise RuntimeError("Training set is empty after merge. Check AS_OF and labels.")

    # Feature engineering
    categorical_cols = [col for col in dataset.columns if dataset[col].dtype == "object" and col not in ["user_id", "event_timestamp"]]
    numerical_cols = [col for col in dataset.columns if col not in categorical_cols + ["user_id", "event_timestamp", "churn_label"]]

    X, y = prepare_dataset(dataframe=dataset)

    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(), categorical_cols),
            ("numerical", "passthrough", numerical_cols),
        ], 
        remainder='drop'
    )

    clf = RandomForestClassifier(
        n_estimators=HYPERPARAMS["n_estimators"],
        n_jobs=HYPERPARAMS["n_jobs"],
        class_weight=HYPERPARAMS['class_weight'],
        max_features=HYPERPARAMS["max_features"],
        random_state=HYPERPARAMS["random_state"]
    )

    pipeline = Pipeline(steps=[("preprocessing", preprocessing), ("clf", clf)])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # mlflow run
    with mlflow.start_run(run_name=f"rf_baseline_{settings.AS_OF}") as run:
        start = time.time()
        pipeline.fit(X=X_train, y=y_train)
        end = time.time()
        training_time = end - start
        if hasattr(pipeline, "predict_proba"):
            y_val_proba = pipeline.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_val_proba)

        y_pred = pipeline.predict(X=X_val)
        metrics = {
            "Accuracy": accuracy_score(y_true=y_val, y_pred=y_pred),
            "precision": precision_score(y_true=y_val, y_pred=y_pred),
            "recall": recall_score(y_true=y_val, y_pred=y_pred),
            "f1-score": f1_score(y_true=y_val, y_pred=y_pred)
        }
        # Metrics logging
        mlflow.log_param(key='hyperparams', value=HYPERPARAMS)
        mlflow.log_metrics(metrics=metrics)
        mlflow.log_dict(
            {"categorical_cols": categorical_cols,
             "numeric_cols": numerical_cols},
            "artifacts/feature_schema.json"
        )

        input_schema = Schema(
            [
                ColSpec("long", "months_active"),
                ColSpec("double", "monthly_fee"),
                ColSpec("boolean", "paperless_billing"),
                ColSpec("boolean", "plan_stream_tv"),
                ColSpec("boolean", "plan_stream_movies"),
                ColSpec("string", "net_service"),
                ColSpec("double", "watch_hours_30d"),
                ColSpec("double", "avg_session_mins_7d"),
                ColSpec("long", "unique_devices_30d"),
                ColSpec("long", "skips_7d"),
                ColSpec("long", "rebuffer_events_7d"),
                ColSpec("long", "failed_payments_90d"),
                ColSpec("long", "support_tickets_90d"),
                ColSpec("double", "ticket_avg_resolution_hrs_90d"),
            ]
        )
        output_schema = Schema([ColSpec("long", "prediction")])
        params_schema = ParamSchema(
            [
                ParamSpec("n_estimators", 'long', HYPERPARAMS["n_estimators"]),
                ParamSpec("n_jobs", 'long', HYPERPARAMS["n_jobs"]),
                ParamSpec("class_weight", 'string', HYPERPARAMS["class_weight"]),
            ]
        )
        signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=params_schema)

        # model logging
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="models",
            registered_model_name=MODEL_NAME,
            signature=signature
        ) 

        print(f"[OK] Trained baseline. AUC={auc:.4f} F1={metrics['f1-score']:.4f} ACC={metrics['Accuracy']:.4f} (run_id={run.info.run_id})")