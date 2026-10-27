"""Operator-issued JWTs; this is not a password, SSO or live identity directory."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .models import Role
from .settings import validate_signing_key

logger = logging.getLogger(__name__)

try:
    import jwt as pyjwt
except ImportError:
    pyjwt = None

DEFAULT_EXPIRY_HOURS = 24
MAX_EXPIRY_HOURS = 24
TOKEN_ISSUER = "forecasting-product"
TOKEN_AUDIENCE = "forecasting-product-api"
REQUIRED_CLAIMS = ("user_id", "email", "role", "is_active", "iat", "exp", "iss", "aud")


def ensure_jwt_available() -> None:
    """Fail startup rather than silently disabling authentication without PyJWT."""
    if pyjwt is None or not callable(getattr(pyjwt, "encode", None)) or not callable(getattr(pyjwt, "decode", None)):
        raise RuntimeError("PyJWT is required for production authentication; install the API requirements")


def _valid_claims(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload.get("user_id"), str) or not payload["user_id"].strip():
        return False
    if not isinstance(payload.get("email"), str) or type(payload.get("is_active")) is not bool:
        return False
    if not isinstance(payload.get("role"), str):
        return False
    try:
        Role(payload["role"])
    except ValueError:
        return False
    if type(payload.get("iat")) is not int or type(payload.get("exp")) is not int:
        return False
    if not 0 < payload["iat"] < payload["exp"]:
        return False
    if payload["exp"] - payload["iat"] > MAX_EXPIRY_HOURS * 3600:
        return False
    if "nbf" in payload and type(payload["nbf"]) is not int:
        return False
    return payload.get("iss") == TOKEN_ISSUER and payload.get("aud") == TOKEN_AUDIENCE


def create_token(
    user_id: str,
    email: str,
    role: str,
    secret_key: str,
    expiry_hours: int = DEFAULT_EXPIRY_HOURS,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a dedicated HS256 token for a trusted operator.

    Additional claims cannot override identity, role, activity, lifetime or
    service scope. Possession of the signing key is the operator trust boundary.
    """
    ensure_jwt_available()
    validate_signing_key(secret_key)
    if type(expiry_hours) is not int or not 1 <= expiry_hours <= MAX_EXPIRY_HOURS:
        raise ValueError("expiry_hours must be an integer from 1 through 24")
    if extra_claims and set(extra_claims).intersection(REQUIRED_CLAIMS):
        raise ValueError("Additional claims cannot replace reserved authentication claims")
    now = int(datetime.now(timezone.utc).timestamp())
    payload = {
        "user_id": user_id,
        "email": email,
        "role": role,
        "is_active": True,
        "iat": now,
        "exp": now + expiry_hours * 3600,
        "iss": TOKEN_ISSUER,
        "aud": TOKEN_AUDIENCE,
        **(extra_claims or {}),
    }
    if not _valid_claims(payload):
        raise ValueError("Token identity and role must be valid; use a nonempty user ID and a supported role")
    return pyjwt.encode(payload, secret_key, algorithm="HS256")


def decode_token(token: str, secret_key: str) -> Optional[Dict[str, Any]]:
    """Verify signature, lifetime, service scope and strictly typed identity claims.

    Invalid credentials return ``None`` with a value-free warning. Signing
    configuration errors are not mistaken for an anonymous/demo identity.
    """
    ensure_jwt_available()
    validate_signing_key(secret_key)
    if not isinstance(token, str) or not token:
        logger.warning("Invalid authentication token")
        return None
    try:
        payload = pyjwt.decode(
            token,
            secret_key,
            algorithms=["HS256"],
            issuer=TOKEN_ISSUER,
            audience=TOKEN_AUDIENCE,
            options={"require": list(REQUIRED_CLAIMS)},
        )
    except (pyjwt.InvalidTokenError, ValueError, TypeError, OverflowError):
        logger.warning("Invalid or expired authentication token")
        return None
    if not _valid_claims(payload):
        logger.warning("Invalid authentication claims")
        return None
    return payload
