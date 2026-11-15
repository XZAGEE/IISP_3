import logging
import os

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from . import crud, schemas
from .auth import (
    create_access_token,
    get_current_user,
    get_password_hash,
    verify_password,
)
from .database import Base, engine, get_db
from .models import User

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("orders-api")

FORECAST_URL = os.getenv(
    "FORECAST_URL",
    "http://forecast-api:8001",
)

FORECAST_TIMEOUT = float(
    os.getenv("FORECAST_TIMEOUT", "3.0")
)

# Tables are created automatically for the initial version.
# In production, replace this with Alembic migrations.
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Orders API",
    version="0.1.0",
)


# -------------------------
# Authentication endpoints
# -------------------------

@app.post(
    "/auth/register",
    response_model=schemas.UserRead,
    status_code=status.HTTP_201_CREATED,
)
def register_user(
    data: schemas.UserCreate,
    db: Session = Depends(get_db),
):
    """Register a new user."""
    existing_user = (
        db.query(User)
        .filter(User.username == data.username)
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already registered.",
        )

    user = User(
        username=data.username,
        hashed_password=get_password_hash(data.password),
    )

    db.add(user)

    try:
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already registered.",
        )

    return user


@app.post("/auth/login")
def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """Authenticate a user and return a JWT access token."""
    user = (
        db.query(User)
        .filter(User.username == form_data.username)
        .first()
    )

    if user is None or not verify_password(
        form_data.password,
        user.hashed_password if user else get_password_hash("dummy-password"),
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(user.username)

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


@app.get(
    "/auth/me",
    response_model=schemas.UserRead,
)
def read_current_user(
    current_user: User = Depends(get_current_user),
):
    """Return information about the authenticated user."""
    return current_user


# -------------------------
# Forecasting helper
# -------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.3, max=2),
    retry=retry_if_exception_type(
        (httpx.ConnectError, httpx.TimeoutException)
    ),
    reraise=True,
)
def _request_prediction(payload: dict) -> float:
    """Request a price prediction from the forecasting service."""
    response = httpx.post(
        f"{FORECAST_URL}/predict",
        json=payload,
        timeout=FORECAST_TIMEOUT,
    )
    response.raise_for_status()

    return float(response.json()["predicted_price"])


# -------------------------
# Public health endpoint
# -------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Return the health status of the Orders API."""
    return {"status": "ok"}


# -------------------------
# Protected order endpoints
# -------------------------

@app.post(
    "/orders",
    response_model=schemas.OrderRead,
    status_code=201,
)
def create_order(
    data: schemas.OrderCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create an order and attempt to obtain a price prediction."""
    try:
        price = _request_prediction(data.model_dump())
    except Exception as exc:
        logger.error(
            "Forecast service is unavailable: %s",
            exc,
        )
        price = None

    return crud.create_order(
        db,
        data,
        price,
    )


@app.get(
    "/orders",
    response_model=list[schemas.OrderRead],
)
def list_orders(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return a paginated list of orders."""
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 100.",
        )

    if offset < 0:
        raise HTTPException(
            status_code=400,
            detail="Offset must be non-negative.",
        )

    return crud.list_orders(
        db,
        limit,
        offset,
    )


@app.get(
    "/orders/{oid}",
    response_model=schemas.OrderRead,
)
def get_order(
    oid: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return a single order by ID."""
    order = crud.get_order(db, oid)

    if order is None:
        raise HTTPException(
            status_code=404,
            detail="Order not found.",
        )

    return order


@app.patch(
    "/orders/{oid}",
    response_model=schemas.OrderRead,
)
def update_order(
    oid: int,
    data: schemas.OrderUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update an existing order."""
    order = crud.get_order(db, oid)

    if order is None:
        raise HTTPException(
            status_code=404,
            detail="Order not found.",
        )

    updated_fields = data.model_dump(exclude_unset=True)

    feature_fields = {
        "gr_liv_area",
        "total_bsmt_sf",
        "garage_area",
        "year_built",
        "overall_qual",
        "ms_zoning",
    }

    if feature_fields.intersection(updated_fields):
        payload = {
            "gr_liv_area": updated_fields.get(
                "gr_liv_area", order.gr_liv_area
            ),
            "total_bsmt_sf": updated_fields.get(
                "total_bsmt_sf", order.total_bsmt_sf
            ),
            "garage_area": updated_fields.get(
                "garage_area", order.garage_area
            ),
            "year_built": updated_fields.get(
                "year_built", order.year_built
            ),
            "overall_qual": updated_fields.get(
                "overall_qual", order.overall_qual
            ),
            "ms_zoning": updated_fields.get(
                "ms_zoning", order.ms_zoning
            ),
        }

        try:
            order.predicted_price = _request_prediction(payload)
        except Exception as exc:
            logger.error(
                "Forecast service is unavailable during order update: %s",
                exc,
            )
            order.predicted_price = None

    return crud.update_order(db, order, data)


@app.delete(
    "/orders/{oid}",
    status_code=204,
)
def delete_order(
    oid: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    """Delete an existing order."""
    order = crud.get_order(db, oid)

    if order is None:
        raise HTTPException(
            status_code=404,
            detail="Order not found.",
        )

    crud.delete_order(db, order)