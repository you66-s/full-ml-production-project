from fastapi import FastAPI
from feast import FeatureStore


app = FastAPI()
store = FeatureStore(repo_path="/repo")
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/features/{user_id}")
def get_features(user_id: str):
    features = [
        "subscription_profile_fv:months_active",
        "subscription_profile_fv:monthly_fee",
        "subscription_profile_fv:paperless_billing",
    ]

    feature_dict = store.get_online_features(
        features=features,
        entity_rows=[{"user_id": user_id}],
    ).to_dict()

    # On convertit en format plus simple (clé -> valeur scalaires)
    simple = {name: values[0] for name, values in feature_dict.items()}

    return {
        "user_id": user_id,
        "features": simple,
    }