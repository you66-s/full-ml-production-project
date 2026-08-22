from pathlib import Path
import pprint
from train_and_promote import train_and_compare_flow
from core.config import get_settings
from core.database import engine
import pandas as pd
from prefect import flow, task
from feast import FeatureStore
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.metrics import ValueDrift
from evidently import Dataset
from evidently import DataDefinition

settings = get_settings()
REPORT_DIR = settings.EVIDENTLY_REPORT_DIR
FEAST_REPO = settings.FEAST_REPO

AS_OF_CUR_DEFAULT = "2024-01-31"
AS_OF_REF_DEFAULT = "2022-01-31"

def fetch_entity_df(engine, as_of: str) -> pd.DataFrame:
    """
    Construit le dataframe des entités nécessaire pour un historical join Feast :
    colonnes obligatoires : user_id + event_timestamp.
    """
    q = """
    SELECT user_id, as_of
    FROM subscriptions_profile_snapshots
    WHERE as_of = %(as_of)s
    """
    df = pd.read_sql(q, engine, params={"as_of": as_of})
    if df.empty:
        raise RuntimeError(
            f"Aucun snapshot trouvé pour as_of={as_of}. "
            "Vérifiez les dates des snapshots dans votre base."
        )
    df = df.rename(columns={"as_of": "event_timestamp"})
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    return df[["user_id", "event_timestamp"]]

def fetch_labels(engine, as_of: str) -> pd.DataFrame:
    """
    Essaie de récupérer des labels alignés temporellement.
    Si un schéma riche existe (labels avec period_start), on filtre sur as_of.
    Sinon, fallback sur (user_id, churn_label) et on injecte un event_timestamp synthétique.
    """
    # Schéma riche
    try:
        q = """
        SELECT user_id, period_start, churn_label
        FROM labels
        WHERE period_start = %(as_of)s
        """
        labels = pd.read_sql(q, engine, params={"as_of": as_of})
        if not labels.empty:
            labels = labels.rename(columns={"period_start": "event_timestamp"})
            labels["event_timestamp"] = pd.to_datetime(labels["event_timestamp"])
            return labels[["user_id", "event_timestamp", "churn_label"]]
    except Exception:
        pass

    # Schéma simple
    q2 = "SELECT user_id, churn_label FROM labels"
    labels = pd.read_sql(q2, engine)
    if labels.empty:
        # Aucun label disponible
        return pd.DataFrame(columns=["user_id", "event_timestamp", "churn_label"])

    labels["event_timestamp"] = pd.to_datetime(as_of)
    return labels[["user_id", "event_timestamp", "churn_label"]]

def build_features(entity_df: pd.DataFrame) -> pd.DataFrame:
    store = FeatureStore(repo_path=FEAST_REPO)

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

    hf = store.get_historical_features(
        entity_df=entity_df,
        features=features,
    )
    return hf.to_df()

def get_final_features(as_of: str) -> pd.DataFrame:
    """
    Construit un dataset final : features Feast + labels si disponibles, alignés sur (user_id, event_timestamp).
    """

    entity_df = fetch_entity_df(engine, as_of)
    feat_df = build_features(entity_df)

    labels_df = fetch_labels(engine, as_of)
    if labels_df.empty:
        # Pas de labels -> on retourne uniquement les features
        return feat_df

    # Merge strict sur user_id + event_timestamp pour éviter toute incohérence temporelle
    df = feat_df.merge(labels_df, on=["user_id", "event_timestamp"], how="inner")
    return df


def build_dataset_from_df(df: pd.DataFrame) -> Dataset:
    """
    Construit un Dataset Evidently avec une définition simple (numérique / catégoriel).
    """
    # Colonnes ignorées (identifiants / timestamps)
    ignored = ["user_id", "event_timestamp"]

    cat_cols = [c for c in df.columns if df[c].dtype in ["object", "bool"] and c not in ignored]
    num_cols = [c for c in df.columns if c not in cat_cols + ignored]

    definition = DataDefinition(
        numerical_columns=num_cols,
        categorical_columns=cat_cols,
    )
    dataset = Dataset.from_pandas(df, data_definition=definition)
    return dataset

@task(name="building dataset")
def build_dataset(as_of: str) -> pd.DataFrame:
    return get_final_features(as_of)


@task(name="drift calculation")
def compute_target_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> float:
    if "churn_label" not in reference_df.columns or "churn_label" not in current_df.columns:
        print("[Target drift] churn_label absent -> target drift non calculé")
        return float("nan")
    if reference_df["churn_label"].dropna().empty or current_df["churn_label"].dropna().empty:
        print("[Target drift] labels vides -> target drift non calculé")
        return float("nan")

    ref_rate = float(reference_df["churn_label"].astype(int).mean())
    cur_rate = float(current_df["churn_label"].astype(int).mean())
    target_drift = abs(cur_rate - ref_rate)
    print(f"[Target drift] ref_rate={ref_rate:.4f} cur_rate={cur_rate:.4f} abs_diff={target_drift:.4f}")
    return target_drift

@task(name="Running evidently task")
def run_evidently(reference_df: pd.DataFrame, current_df: pd.DataFrame, as_of_ref: str, as_of_cur: str):
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

    DRIFT_SHARE_THRESHOLD = 0.2

    # Evidently : on combine un résumé + une détection de drift + un drift sur une colonne (si présente)
    metrics = [
        DataSummaryPreset(),
        DataDriftPreset(drift_share=DRIFT_SHARE_THRESHOLD),
        ValueDrift(column="churn_label"),
    ]

    report = Report(metrics=metrics)

    eval_result = report.run(
        reference_data=build_dataset_from_df(reference_df),
        current_data=build_dataset_from_df(current_df),
    )

    html_path = Path(REPORT_DIR) / f"drift_{as_of_ref}_vs_{as_of_cur}.html"
    json_path = Path(REPORT_DIR) / f"drift_{as_of_ref}_vs_{as_of_cur}.json"
    eval_result.save_html(str(html_path))
    eval_result.save_json(str(json_path))

    summary = eval_result.dict()
    pprint.pprint(summary)

    # Signal scalaire : part de features driftées (si trouvé)
    drift_share = None
    for metric in summary.get("metrics", []):
        if "DriftedColumnsCount" in metric.get("metric_id", ""):
            drift_share = metric["value"]["share"]

    if drift_share is None:
        # Fallback si Evidently change légèrement la structure : on garde une valeur neutre
        drift_share = 0.0

    return {
        "html": str(html_path),
        "json": str(json_path),
        "drift_share": float(drift_share),
    }

@task(name="task to decide")
def decide_action(as_of_ref: str, as_of_cur: str, drift_share: float, target_drift: float, threshold: float = 0.3) -> str:
    """
    Décision simple : si drift_share dépasse threshold, on simule un déclenchement de retrain.
    """
    if drift_share >= threshold:
        return (
            f"RETRAINING_TRIGGERED drift_share={drift_share:.2f} >= {threshold:.2f} "
            f"(target_drift={target_drift if target_drift == target_drift else 'NaN'})"
        )
    return (
        f"NO_ACTION drift_share={drift_share:.2f} < {threshold:.2f} "
        f"(target_drift={target_drift if target_drift == target_drift else 'NaN'})"
    )

@flow(name="monitor_month")
def monitor_month_flow(as_of_ref: str = AS_OF_REF_DEFAULT,as_of_cur: str = AS_OF_CUR_DEFAULT,threshold: float = 0.3,):
    ref_df = build_dataset(as_of_ref)
    cur_df = build_dataset(as_of_cur)

    # drift sur la cible si les labels sont disponibles
    tdrift = compute_target_drift(ref_df, cur_df)

    res = run_evidently(ref_df, cur_df, as_of_ref, as_of_cur)
    msg = decide_action(as_of_ref, as_of_cur, res["drift_share"], tdrift, threshold)

    print(
        f"[Evidently] report_html={res['html']} report_json={res['json']} "
        f"drift_share={res['drift_share']:.2f} -> {msg}"
    )

@task(name="running task of retraining decision")
def retrain_launcher(drift_share: float, target_drift: float, threshold: float = 0.00):
    if drift_share >= threshold:
        decision = train_and_compare_flow()
        return f"RETRAINING_TRIGGERED drift_share={drift_share:.2f} >= {threshold:.2f} -> {decision}"
    return f"NO_ACTION drift_share={drift_share:.2f} < {threshold:.2f}"
if __name__ == "__main__":
    monitor_month_flow()