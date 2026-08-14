from core.config import get_settings
from core.database import engine
import pandas as pd
from feast import FeatureStore
import os

AS_OF = "2024-01-31"
FEAST_REPO = "/repo"

def build_entity_df(engine, as_of: str) -> pd.DataFrame:
    query = "SELECT user_id, as_of FROM subscriptions_profile_snapshots where as_of = %(as_of)s"
    dataframe = pd.read_sql(query, engine, params={"as_of": as_of})
    if dataframe.empty:
        raise RuntimeError(f"No snapshot rows found at as_of={as_of}")
    dataframe = dataframe.rename(columns={"as_of": "event_timestamp"})
    dataframe["event_timestamp"] = pd.to_datetime(dataframe["event_timestamp"])
    return dataframe[["user_id", "event_timestamp"]]

def fetch_labels(engine, as_of: str) -> pd.DataFrame:
    query = "SELECT user_id, churn_label FROM labels"
    labels = pd.read_sql(query, engine)
    if labels.empty:
        raise RuntimeError("Labels table is empty.")
    labels["event_timestamp"] = pd.to_datetime(as_of)
    return labels[["user_id", "event_timestamp", "churn_label"]]

if __name__ == "__main__":
    entity_df = build_entity_df(engine, AS_OF)
    labels = fetch_labels(engine, AS_OF)
    store = FeatureStore(repo_path=FEAST_REPO)

    # la liste de features à récupérer pour la creation du dataset
    features = [
        "subscription_profile_fv:months_active",
        "subscription_profile_fv:monthly_fee",
        "subscription_profile_fv:paperless_billing",
        "usage_agg_30d_fv:watch_hours_30d",
        "usage_agg_30d_fv:avg_session_mins_7d",
        "payments_agg_90d_fv:failed_payments_90d",
    ]

    histo_features = store.get_historical_features(
        entity_df=entity_df,
        features=features
    ).to_df()

    merged_dataframe = histo_features.merge(labels, on=["user_id", "event_timestamp"], how="inner")
    if merged_dataframe.empty:

        raise RuntimeError("Training set is empty after merge. Check AS_OF and labels.")
    os.makedirs("/data/processed", exist_ok=True)
    merged_dataframe.to_csv("/data/processed/training_df.csv", index=False)
    print(f"[OK] Wrote /data/processed/training_df.csv with {len(merged_dataframe)} rows")