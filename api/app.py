from fastapi import FastAPI, HTTPException, status, Response
from core.config import get_settings
from feast import FeatureStore
from schemes import UserPayload
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import pandas as pd
import mlflow.pyfunc, time

app = FastAPI(title="Churn Prediction API")
settings = get_settings()

mlflow.set_tracking_uri("http://mlflow:5000")
MODEL_URI = "models:/churn_v1@production"
try:
    store = FeatureStore(repo_path="/repo")
    model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception as e:
    print(f"Warning: model initialization failed: {e}")
    store = None
    model = None

REQUEST_COUNT = Counter(name="api_requests_total", documentation="Total number of API requests")
REQUEST_LATENCY = Histogram(name="api_request_latency_seconds", documentation="Latency of API requests in seconds")
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: UserPayload):
    start = time.time()
    REQUEST_COUNT.inc()
    if store is None or model is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model/store not initialized")
    features_request = [
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

    online_features = store.get_online_features(
        features=features_request,
        entity_rows=[{"user_id": payload.user_id}],
    ).to_dict()

    X = pd.DataFrame({k: [v[0]] for k, v in online_features.items()})

    # Gestion des features manquantes
    if X.isnull().any().any():
        missing = X.columns[X.isnull().any()].tolist()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)
    # Nettoyage minimal (évite bugs de types)
    X = X.drop(columns=["user_id"], errors="ignore")
    y_pred = model.predict(data=X)
    
    end = time.time() - start
    REQUEST_LATENCY.observe(end)
    return {
        "user_id": payload.user_id,
        "prediction": int(y_pred[0]),
        "features_used": X.to_dict(orient="records")[0],
    }


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)