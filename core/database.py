from sqlalchemy import create_engine, URL
from sqlalchemy.orm import sessionmaker
from core.config import get_settings

settings = get_settings()

url = URL.create(
    drivername="postgresql+psycopg2",
    username=settings.POSTGRES_USER,
    password=settings.POSTGRES_PASSWORD,
    host=settings.POSTGRES_HOST,
    database=settings.POSTGRES_DB,
    port=settings.POSTGRES_PORT
)

engine = create_engine(url=url, pool_pre_ping=True, pool_size=10, max_overflow=10)

session = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)