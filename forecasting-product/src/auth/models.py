"""
User and role models for RBAC.

Defines the role hierarchy and user representation used throughout
the platform for access control and audit attribution.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Set


class Role(str, Enum):
    """Platform roles — ordered by decreasing privilege."""
    ADMIN = "admin"
    DATA_SCIENTIST = "data_scientist"
    PLANNER = "planner"
    MANAGER = "manager"
    VIEWER = "viewer"


class Permission(str, Enum):
    """Fine-grained permissions checked by RBAC middleware."""
    VIEW_FORECASTS = "view_forecasts"
    VIEW_METRICS = "view_metrics"
    VIEW_AUDIT_LOG = "view_audit_log"
    CREATE_OVERRIDE = "create_override"
    DELETE_OVERRIDE = "delete_override"
    APPROVE_OVERRIDE = "approve_override"
    RUN_BACKTEST = "run_backtest"
    RUN_PIPELINE = "run_pipeline"
    PROMOTE_MODEL = "promote_model"
    MODIFY_CONFIG = "modify_config"
    MANAGE_USERS = "manage_users"


# Role → permissions mapping
ROLE_PERMISSIONS: dict[Role, Set[Permission]] = {
    Role.ADMIN: set(Permission),  # all permissions
    Role.DATA_SCIENTIST: {
        Permission.VIEW_FORECASTS,
        Permission.VIEW_METRICS,
        Permission.VIEW_AUDIT_LOG,
        Permission.CREATE_OVERRIDE,
        Permission.DELETE_OVERRIDE,
        Permission.RUN_BACKTEST,
        Permission.RUN_PIPELINE,
        Permission.PROMOTE_MODEL,
        Permission.MODIFY_CONFIG,
    },
    Role.PLANNER: {
        Permission.VIEW_FORECASTS,
        Permission.VIEW_METRICS,
        Permission.CREATE_OVERRIDE,
    },
    Role.MANAGER: {
        Permission.VIEW_FORECASTS,
        Permission.VIEW_METRICS,
        Permission.VIEW_AUDIT_LOG,
        Permission.APPROVE_OVERRIDE,
    },
    Role.VIEWER: {
        Permission.VIEW_FORECASTS,
        Permission.VIEW_METRICS,
    },
}


@dataclass
class User:
    """Represents an authenticated platform user."""
    user_id: str
    email: str
    role: Role
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    expires_at: Optional[int] = None

    def has_permission(self, permission: Permission) -> bool:
        """Check if user's role grants the given permission."""
        if not self.is_active:
            return False
        return permission in ROLE_PERMISSIONS.get(self.role, set())

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "role": self.role.value,
            "is_active": self.is_active,
        }
