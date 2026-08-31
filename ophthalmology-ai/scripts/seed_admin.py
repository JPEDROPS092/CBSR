#!/usr/bin/env python3
"""Create the first administrator.

Usage::

    python scripts/seed_admin.py --email admin@clinic.org --name "Admin"

The password is read from ``ADMIN_PASSWORD`` or prompted for interactively -
never passed on the command line, where it would land in the shell history and
the process table.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

from app.core.enums import UserRole
from app.core.exceptions import AppError
from app.core.logging import configure_logging
from app.database.session import session_scope
from app.services.auth_service import AuthService


def main() -> int:
    """Create an admin user; returns a process exit code."""
    parser = argparse.ArgumentParser(description="Create an administrator account.")
    parser.add_argument("--email", required=True)
    parser.add_argument("--name", default="Administrator")
    parser.add_argument(
        "--role",
        default=UserRole.ADMIN,
        choices=[str(role) for role in UserRole],
        help="Role to assign (default: admin).",
    )
    args = parser.parse_args()

    configure_logging("INFO", "console")
    password = os.getenv("ADMIN_PASSWORD") or getpass.getpass("Password: ")

    try:
        with session_scope() as session:
            user = AuthService(session).create_user(
                email=args.email,
                full_name=args.name,
                password=password,
                role=UserRole(args.role),
            )
            print(f"Created {user.role} user {user.email} ({user.id})")
    except AppError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
