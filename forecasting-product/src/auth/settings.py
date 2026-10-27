"""Fail-closed authentication settings shared by API factories and launchers."""

import os
from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional


def _boolean_setting(
    name: str,
    explicit: Optional[bool],
    environment: Mapping[str, str],
    default: bool,
) -> bool:
    if explicit is not None:
        if type(explicit) is not bool:
            raise ValueError(f"{name} must be a boolean")
        return explicit
    value = environment.get(name)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{name} must be true, false, 1, or 0")
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"{name} must be true, false, 1, or 0")


def validate_signing_key(secret_key: str) -> None:
    """Reject missing, short, whitespace, placeholder and trivially weak keys.

    Operators must generate a dedicated key from at least 32 random bytes;
    configuration checks cannot prove the entropy of an arbitrary string.
    """
    placeholders = (
        "change-me", "changeme", "change_me", "your-secret", "replace-me",
        "replace-with", "docker-dev", "development-secret", "test-secret",
    )
    if (
        not isinstance(secret_key, str)
        or not secret_key.isprintable()
        or any(character.isspace() for character in secret_key)
        or len(secret_key.encode("utf-8")) < 32
        or len(set(secret_key)) < 8
        or secret_key.lower().startswith(placeholders)
    ):
        raise ValueError(
            "JWT_SECRET_KEY must be configured from at least 32 random bytes; "
            "blank, short, whitespace, placeholder and trivially repeated keys are not allowed"
        )


@dataclass(frozen=True)
class AuthSettings:
    """A production JWT boundary or an explicitly unauthenticated demo."""

    auth_enabled: bool = True
    dev_mode: bool = False
    jwt_secret: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if type(self.auth_enabled) is not bool or type(self.dev_mode) is not bool:
            raise ValueError("AUTH_ENABLED and API_DEV_MODE must be booleans")
        if not isinstance(self.jwt_secret, str):
            raise ValueError("JWT_SECRET_KEY must be a string")
        if self.auth_enabled == self.dev_mode:
            raise ValueError(
                "Production requires AUTH_ENABLED=true and API_DEV_MODE=false; "
                "explicit demo requires AUTH_ENABLED=false and API_DEV_MODE=true"
            )
        if self.dev_mode:
            if self.jwt_secret:
                raise ValueError(
                    "Explicit demo must leave JWT_SECRET_KEY unset; "
                    "do not give a demo production signing credentials"
                )
        else:
            validate_signing_key(self.jwt_secret)
            from .token import ensure_jwt_available
            ensure_jwt_available()

    @property
    def mode(self) -> Literal["production", "demo"]:
        """Return the public mode name, never signing configuration."""
        return "demo" if self.dev_mode else "production"

    @classmethod
    def from_env(
        cls,
        *,
        auth_enabled: Optional[bool] = None,
        dev_mode: Optional[bool] = None,
        jwt_secret: Optional[str] = None,
        environment: Optional[Mapping[str, str]] = None,
    ) -> "AuthSettings":
        """Resolve explicit factory arguments before strictly parsed environment values."""
        env = os.environ if environment is None else environment
        return cls(
            auth_enabled=_boolean_setting("AUTH_ENABLED", auth_enabled, env, True),
            dev_mode=_boolean_setting("API_DEV_MODE", dev_mode, env, False),
            jwt_secret=env.get("JWT_SECRET_KEY", "") if jwt_secret is None else jwt_secret,
        )
