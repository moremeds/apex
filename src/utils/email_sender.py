"""Shared email sender — reads SMTP config from secrets.yaml with env var overrides.

Used by momentum, PEAD, and signal runners to send email notifications.
Fail-open: returns False on error, never raises.

Config priority:
    1. Environment variables (CI: SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SIGNAL_REPORT_RECIPIENTS)
    2. config/secrets.yaml (local dev)
"""

from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_secrets_yaml() -> dict[str, Any]:
    """Load config/secrets.yaml if it exists."""
    path = _PROJECT_ROOT / "config" / "secrets.yaml"
    if not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text()) or {}
    except Exception as e:
        logger.warning(f"Failed to read secrets.yaml: {e}")
        return {}


def _get_smtp_config() -> dict[str, str]:
    """Resolve SMTP config: env vars take precedence over secrets.yaml."""
    secrets = _load_secrets_yaml()
    email_cfg = secrets.get("email", {})

    return {
        "server": os.environ.get("SMTP_SERVER", "smtp.gmail.com"),
        "port": os.environ.get("SMTP_PORT", "587"),
        "username": os.environ.get("SMTP_USERNAME", email_cfg.get("SMTP_USERNAME", "")),
        "password": os.environ.get("SMTP_PASSWORD", email_cfg.get("SMTP_APP_PASSWORD", "")),
        "recipients": os.environ.get("SIGNAL_REPORT_RECIPIENTS", email_cfg.get("EMAIL_DEST", "")),
    }


def send_email(
    subject: str,
    body: str,
    content_type: str = "text/plain",
) -> bool:
    """Send an email via SMTP.

    Args:
        subject: Email subject line.
        body: Email body text.
        content_type: MIME content type (default: text/plain).

    Returns:
        True if sent successfully, False otherwise.
    """
    cfg = _get_smtp_config()

    if not cfg["username"] or not cfg["password"]:
        logger.warning("SMTP credentials not configured. Skipping email.")
        return False

    if not cfg["recipients"]:
        logger.warning("No email recipients configured. Skipping email.")
        return False

    try:
        msg = MIMEText(body, content_type.split("/")[-1])
        msg["Subject"] = subject
        msg["From"] = f"APEX Signal Bot <{cfg['username']}>"
        msg["To"] = cfg["recipients"]

        with smtplib.SMTP(cfg["server"], int(cfg["port"]), timeout=30) as server:
            server.starttls()
            server.login(cfg["username"], cfg["password"])
            server.send_message(msg)

        logger.info(f"Email sent: {subject}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False
