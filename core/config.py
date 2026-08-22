from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: str
    
    AS_OF: str
    SEED_DIR: str

    MLFLOW_TRACKING_URI: str
    MLFLOW_EXPERIMENT: str
    
    EVIDENTLY_REPORT_DIR: str
    FEAST_REPO: str
    
    class Config:
        env_file = ".env"
    
def get_settings():
    return Settings()        