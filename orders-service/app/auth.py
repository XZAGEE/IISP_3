import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pwdlib import PasswordHash
from sqlalchemy.orm import Session

from .database import get_db, read_secret
from .models import User


# Read JWT secret exclusively from Docker Secret.
JWT_SECRET_KEY = read_secret("jwt_secret_key")

# Token lifetime in minutes (default: 30).
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
)

if ACCESS_TOKEN_EXPIRE_MINUTES <= 0:
    raise RuntimeError("ACCESS_TOKEN_EXPIRE_MINUTES must be positive.")


# Argon2id password hashing.
password_hash = PasswordHash.recommended()


# OAuth2 configuration for Swagger UI.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_password_hash(password: str) -> str:
    """Hash a password using Argon2id."""
    return password_hash.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its stored hash."""
    return password_hash.verify(plain_password, hashed_password)


def create_access_token(username: str) -> str:
    """Create a signed JWT access token."""
    now = datetime.now(timezone.utc)

    payload = {
        "sub": username,
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }

    return jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm="HS256",
    )


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Decode JWT and return the authenticated active user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=["HS256"],
        )

        username = payload.get("sub")

        if not isinstance(username, str) or not username:
            raise credentials_exception

    except jwt.InvalidTokenError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()

    if user is None or not user.is_active:
        raise credentials_exception

    return user