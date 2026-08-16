from fastapi import FastAPI
from feast import FeatureStore
from api.schemes import UserPayload
import pandas as pd
import mlflow.pyfunc

app = FastAPI(title="Churn Prediction API")
MODEL_URI = "models:/churn_v1/Production"
try:
    store = FeatureStore(repo_path="/repo")
    model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception as e:
    print(f"Warning: model initialization failed: {e}")
    store = None
    model = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: UserPayload):
    if store is None or model is None:
        return {"error": "Model or feature store not initialized"}
    features_request = [
            "subs_profile_fv:months_active",
            "subs_profile_fv:monthly_fee",
            "subs_profile_fv:paperless_billing",
            "subs_profile_fv:plan_stream_tv",
            "subs_profile_fv:plan_stream_movies",
            "subs_profile_fv:net_service",
            "usage_agg_30d_fv:watch_hours_30d",
            "usage_agg_30d_fv:avg_session_mins_7d",
            "usage_agg_30d_fv:unique_devices_30d",
            "usage_agg_30d_fv:skips_7d",
            "usage_agg_30d_fv:rebuffer_events_7d",
            "payments_agg_90d_fv:failed_payments_90d",
            "support_agg_90d_fv:support_tickets_90d",
            "support_agg_90d_fv:ticket_avg_resolution_hrs_90d",
        ]

    online_features = store.get_online_features(
        features=features_request,
        entity_rows=[{"user_id": payload.user_id}],
    ).to_dict()

    X = pd.DataFrame({k: [v[0]] for k, v in online_features.items()})

    # Gestion des features manquantes
    if X.isnull().any().any():
        missing = X.columns[X.isnull().any()].tolist()
        return {
            "error": f"Missing features for user_id={payload.user_id}",
            "missing_features": missing,
        }
    # Nettoyage minimal (évite bugs de types)
    X = X.drop(columns=["user_id"], errors="ignore")
    y_pred = model.predict(data=X)

    return {
        "user_id": payload.user_id,
        "prediction": int(y_pred),
        "features_used": X.to_dict(orient="records")[0],
    }