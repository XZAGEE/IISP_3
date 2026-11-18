import logging
import os
import httpx
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Response,
    status,
)
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
from .database import Base, engine, get_db, SessionLocal
from .models import User
import asyncio

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
BATCH_MAX_ITEMS = int(
    os.getenv("BATCH_MAX_ITEMS", "50")
)

# Tables are created automatically for the initial version.
# In production, replace this with Alembic migrations.
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Orders API",
    version="0.1.0",
)

@app.on_event("startup")
async def startup_event():
    """Create a shared HTTP client on application startup."""
    app.state.http_client = httpx.AsyncClient(
        timeout=FORECAST_TIMEOUT
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Close the shared HTTP client on application shutdown."""
    await app.state.http_client.aclose()

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

async def _request_batch_prediction(
    client: httpx.AsyncClient,
    payload: dict,
) -> float:
    """Request a price prediction asynchronously."""
    response = await client.post(
        f"{FORECAST_URL}/predict",
        json=payload,
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
# Background prediction
# -------------------------

def predict_order_in_background(
    order_id: int,
    payload: dict,
) -> None:
    """Calculate and save an order prediction in the background."""
    from datetime import datetime, timezone

    db = SessionLocal()

    try:
        price = _request_prediction(payload)

        updated_order = crud.update_order_prediction(
            db,
            order_id,
            price,
        )

        if updated_order is not None:
            logger.info(
                "Background prediction completed: order_id=%s, "
                "price=%.2f, timestamp=%s",
                order_id,
                price,
                datetime.now(timezone.utc).isoformat(),
            )

    except Exception:
        logger.exception(
            "Background prediction failed: order_id=%s",
            order_id,
        )

        crud.update_order_prediction(
            db,
            order_id,
            None,
        )

    finally:
        db.close()

# -------------------------
# Batch order endpoint
# -------------------------

@app.post("/orders/batch")
async def create_orders_batch(
    data: list[schemas.OrderCreate],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create multiple orders and request predictions concurrently."""

    if len(data) == 0 or len(data) > BATCH_MAX_ITEMS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch must contain between 1 and {BATCH_MAX_ITEMS} orders.",
        )

    client = app.state.http_client

    async def get_prediction(index: int, order_data: schemas.OrderCreate):
        try:
            price = await _request_batch_prediction(
                client,
                order_data.model_dump(),
            )
            return {
                "index": index,
                "price": price,
                "error": None,
            }

        except Exception as exc:
            logger.exception(
                "Batch prediction failed for item %s",
                index,
            )
            return {
                "index": index,
                "price": None,
                "error": str(exc),
            }

    # Request all predictions concurrently.
    predictions = await asyncio.gather(
        *[
            get_prediction(index, order_data)
            for index, order_data in enumerate(data)
        ]
    )

    results = []

    # Save orders independently using the database session.
    for prediction, order_data in zip(predictions, data):
        try:
            order = crud.create_order(
                db,
                order_data,
                prediction["price"],
            )

            if prediction["error"] is not None:
                crud.update_order_prediction(
                    db,
                    order.id,
                    None,
                )

            result = {
                "index": prediction["index"],
                "order_id": order.id,
                "status": (
                    "failed"
                    if prediction["error"] is not None
                    else "priced"
                ),
            }

            if prediction["error"] is not None:
                result["error"] = prediction["error"]
            else:
                result["predicted_price"] = prediction["price"]

            results.append(result)

        except Exception:
            db.rollback()
            logger.exception(
                "Could not save batch item %s",
                prediction["index"],
            )

            results.append({
                "index": prediction["index"],
                "order_id": None,
                "status": "error",
                "error": "Could not save order.",
            })

    return {
        "total": len(results),
        "results": results,
    }

# -------------------------
# Protected order endpoints
# -------------------------

@app.post(
    "/orders",
    response_model=schemas.OrderRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_order(
    data: schemas.OrderCreate,
    background_tasks: BackgroundTasks,
    response: Response,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create an order and schedule background price prediction."""
    order = crud.create_order(
        db,
        data,
        None,
    )

    background_tasks.add_task(
        predict_order_in_background,
        order.id,
        data.model_dump(),
    )

    response.headers["Location"] = f"/orders/{order.id}"

    logger.info(
        "Order created: order_id=%s, status=pending",
        order.id,
    )

    return order


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
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update an order and schedule a new prediction if needed."""
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

    property_changed = bool(
        feature_fields.intersection(updated_fields)
    )

    # Update the provided order fields.
    order = crud.update_order(db, order, data)

    if property_changed:
        # Clear the old prediction and mark the order as pending.
        order.predicted_price = None
        order.status = "pending"
        db.commit()
        db.refresh(order)

        # Build the payload using the updated property features.
        payload = {
            "gr_liv_area": order.gr_liv_area,
            "total_bsmt_sf": order.total_bsmt_sf,
            "garage_area": order.garage_area,
            "year_built": order.year_built,
            "overall_qual": order.overall_qual,
            "ms_zoning": order.ms_zoning,
        }

        background_tasks.add_task(
            predict_order_in_background,
            order.id,
            payload,
        )

        logger.info(
            "Order updated: order_id=%s, status=pending",
            order.id,
        )

    return order


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