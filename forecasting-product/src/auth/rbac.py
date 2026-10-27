"""Fail-closed JWT authentication and existing role/permission enforcement."""

import logging
from typing import Any, Callable, Coroutine, Optional
from urllib.parse import unquote

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .models import Permission, Role, User
from .settings import AuthSettings
from .token import decode_token

logger = logging.getLogger(__name__)
_security = HTTPBearer(auto_error=False)


def get_auth_settings(request: Request) -> AuthSettings:
    """Reject an unconfigured app rather than inferring anonymous-admin mode."""
    settings = getattr(request.app.state, "auth_settings", None)
    if not isinstance(settings, AuthSettings):
        logger.error("Authentication settings are not initialized")
        raise HTTPException(503, detail="Authentication is not configured.")
    return settings


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
) -> User:
    """Return a verified production identity or an explicitly simulated demo user."""
    settings = get_auth_settings(request)
    if settings.dev_mode:
        if "authorization" in request.headers:
            raise HTTPException(401, detail="Do not send production credentials to a demo API.")
        try:
            role = Role(request.headers.get("X-Demo-Role", "admin"))
        except ValueError:
            raise HTTPException(401, detail="Invalid demo role.") from None
        try:
            username = unquote(request.headers.get("X-Demo-User", "system"), errors="strict").strip()
        except UnicodeError:
            raise HTTPException(401, detail="Invalid demo user name.") from None
        if not username or len(username) > 128 or not username.isprintable():
            raise HTTPException(401, detail="A demo user name is required.")
        return User(user_id=username, email="", role=role)

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(credentials.credentials, settings.jwt_secret)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not payload["is_active"]:
        raise HTTPException(403, detail="User account is deactivated.")
    return User(
        user_id=payload["user_id"],
        email=payload["email"],
        role=Role(payload["role"]),
        is_active=True,
        expires_at=payload["exp"],
    )


def require_permission(permission: Permission) -> Callable[..., Coroutine[Any, Any, User]]:
    """Require one of the existing role-derived permissions."""
    async def _check(user: User = Depends(get_current_user)) -> User:
        if not user.has_permission(permission):
            logger.warning(
                "Permission denied: role=%s permission=%s",
                user.role.value, permission.value,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission.value}' required. "
                       f"Your role '{user.role.value}' does not have this permission.",
            )
        return user

    return _check


def require_role(*roles: Role) -> Callable[..., Coroutine[Any, Any, User]]:
    """Require one of the specified roles."""
    async def _check(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            role_names = ", ".join(r.value for r in roles)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of roles [{role_names}] required. "
                       f"Your role: '{user.role.value}'.",
            )
        return user

    return _check
