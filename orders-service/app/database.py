
import os
from pathlib import Path

from sqlalchemy import URL, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


def read_secret(name: str) -> str:
    """Read a secret from Docker secrets directory."""
    secret_path = Path("/run/secrets") / name

    try:
        secret = secret_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(
            f"Required secret '{name}' is missing or cannot be read."
        ) from exc

    if not secret:
        raise RuntimeError(f"Required secret '{name}' is empty.")

    return secret


# Read database password from Docker Secret
db_password = read_secret("db_password")

# Read non-sensitive database configuration
db_user = os.getenv("POSTGRES_USER")
db_name = os.getenv("POSTGRES_DB")

if not db_user or not db_name:
    raise RuntimeError("POSTGRES_USER and POSTGRES_DB must be configured.")

# Build database URL safely
DATABASE_URL = URL.create(
    drivername="postgresql+psycopg",
    username=db_user,
    password=db_password,
    host="postgres",
    port=5432,
    database=db_name,
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
