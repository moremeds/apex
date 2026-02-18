"""Tests for src.utils.email_sender."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.utils.email_sender import _get_smtp_config, send_email


class TestGetSmtpConfig:
    """Test SMTP config resolution."""

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SMTP_SERVER", "custom.smtp.com")
        monkeypatch.setenv("SMTP_PORT", "465")
        monkeypatch.setenv("SMTP_USERNAME", "user@test.com")
        monkeypatch.setenv("SMTP_PASSWORD", "secret")
        monkeypatch.setenv("SIGNAL_REPORT_RECIPIENTS", "dest@test.com")

        cfg = _get_smtp_config()
        assert cfg["server"] == "custom.smtp.com"
        assert cfg["port"] == "465"
        assert cfg["username"] == "user@test.com"
        assert cfg["password"] == "secret"
        assert cfg["recipients"] == "dest@test.com"

    def test_defaults_without_env_or_yaml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear env vars that might be set
        for key in [
            "SMTP_SERVER",
            "SMTP_PORT",
            "SMTP_USERNAME",
            "SMTP_PASSWORD",
            "SIGNAL_REPORT_RECIPIENTS",
        ]:
            monkeypatch.delenv(key, raising=False)

        with patch("src.utils.email_sender._load_secrets_yaml", return_value={}):
            cfg = _get_smtp_config()
            assert cfg["server"] == "smtp.gmail.com"
            assert cfg["port"] == "587"
            assert cfg["username"] == ""
            assert cfg["password"] == ""

    def test_load_from_secrets_yaml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in [
            "SMTP_SERVER",
            "SMTP_PORT",
            "SMTP_USERNAME",
            "SMTP_PASSWORD",
            "SIGNAL_REPORT_RECIPIENTS",
        ]:
            monkeypatch.delenv(key, raising=False)

        mock_yaml = {
            "email": {
                "SMTP_USERNAME": "yaml@test.com",
                "SMTP_APP_PASSWORD": "yaml-pass",
                "EMAIL_DEST": "yaml-dest@test.com",
            }
        }
        with patch("src.utils.email_sender._load_secrets_yaml", return_value=mock_yaml):
            cfg = _get_smtp_config()
            assert cfg["username"] == "yaml@test.com"
            assert cfg["password"] == "yaml-pass"
            assert cfg["recipients"] == "yaml-dest@test.com"


class TestSendEmail:
    """Test send_email() function."""

    def test_returns_false_without_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in ["SMTP_USERNAME", "SMTP_PASSWORD", "SIGNAL_REPORT_RECIPIENTS"]:
            monkeypatch.delenv(key, raising=False)

        with patch("src.utils.email_sender._load_secrets_yaml", return_value={}):
            result = send_email("Test Subject", "Test Body")
            assert result is False

    def test_returns_false_on_smtp_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SMTP_USERNAME", "user@test.com")
        monkeypatch.setenv("SMTP_PASSWORD", "secret")
        monkeypatch.setenv("SIGNAL_REPORT_RECIPIENTS", "dest@test.com")

        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)
        mock_smtp.starttls.side_effect = ConnectionRefusedError("Connection refused")

        with patch("src.utils.email_sender.smtplib.SMTP", return_value=mock_smtp):
            result = send_email("Test", "Body")
            assert result is False

    def test_returns_true_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SMTP_USERNAME", "user@test.com")
        monkeypatch.setenv("SMTP_PASSWORD", "secret")
        monkeypatch.setenv("SIGNAL_REPORT_RECIPIENTS", "dest@test.com")

        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)

        with patch("src.utils.email_sender.smtplib.SMTP", return_value=mock_smtp):
            result = send_email("Test Subject", "Test Body")
            assert result is True
            mock_smtp.starttls.assert_called_once()
            mock_smtp.login.assert_called_once_with("user@test.com", "secret")
            mock_smtp.send_message.assert_called_once()
