"""Nonsensitive mode discovery and server-verified identity; no HTTP token issuer."""

from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from ...auth.models import ROLE_PERMISSIONS, Permission, Role, User
from ...auth.rbac import get_auth_settings, get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


class AuthStatusResponse(BaseModel):
    mode: Literal["production", "demo"]
    auth_required: bool


class AuthIdentityResponse(BaseModel):
    mode: Literal["production", "demo"]
    username: str
    email: str
    role: Role
    permissions: List[Permission]
    expires_at: Optional[int]


@router.get("/status", response_model=AuthStatusResponse)
def auth_status(request: Request) -> AuthStatusResponse:
    """Expose mode only, never key material or signing configuration."""
    settings = get_auth_settings(request)
    return AuthStatusResponse(mode=settings.mode, auth_required=settings.auth_enabled)


@router.get("/me", response_model=AuthIdentityResponse)
def current_identity(
    request: Request, user: User = Depends(get_current_user),
) -> AuthIdentityResponse:
    """Derive identity and permissions on the server, not from client-selected roles."""
    return AuthIdentityResponse(
        mode=get_auth_settings(request).mode,
        username=user.user_id,
        email=user.email,
        role=user.role,
        permissions=sorted(ROLE_PERMISSIONS[user.role], key=lambda item: item.value),
        expires_at=user.expires_at,
    )
