"""Provision a short-lived operator JWT to a new restricted file, never stdout."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auth.issuer import write_private_token
from src.auth.models import Role
from src.auth.settings import AuthSettings
from src.auth.token import create_token


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trusted operator token provisioning; signing configuration comes from the environment",
    )
    parser.add_argument("--user-id", required=True, help="Operator-assigned identity, not an account lookup")
    parser.add_argument("--email", default="", help="Optional identity label")
    parser.add_argument("--role", required=True, choices=[role.value for role in Role])
    parser.add_argument("--expiry-hours", type=int, default=1, help="Lifetime from 1 to 24 hours; default 1")
    parser.add_argument("--output", required=True, type=Path, help="New owner-only file in an existing protected directory")
    args = parser.parse_args()
    try:
        settings = AuthSettings.from_env()
        if settings.dev_mode:
            raise ValueError("Operator token issuance requires production authentication configuration")
        token = create_token(
            user_id=args.user_id, email=args.email, role=args.role,
            secret_key=settings.jwt_secret, expiry_hours=args.expiry_hours,
        )
        write_private_token(args.output, token)
    except (ValueError, RuntimeError, OSError):
        parser.exit(
            2,
            "Token issuance failed. Check production signing configuration, identity/role, "
            "1-24 hour lifetime, and a new output file in an owner-only-capable directory. "
            "An empty or private partial file may remain; no credential was printed.\n",
        )
    print("Token written to the requested restricted file; keep it out of source control.")


if __name__ == "__main__":
    main()
