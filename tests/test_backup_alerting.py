"""Tests for database backup Slack alerting integration."""
import pytest
from unittest.mock import patch, MagicMock

from backend.middleware.alerting import alert_backup_status


class TestBackupAlerting:
    """Verify alert_backup_status is called at the correct backup outcome points."""

    @patch("backend.middleware.alerting.alert_backup_status")
    def test_not_postgres_dev_no_alert(self, mock_alert):
        """SQLite DB + ENVIRONMENT=development → no alert (expected dev-mode skip)."""
        with patch("backend.tasks.scheduled_tasks.DATABASE_URL", "sqlite:///dev.db"), \
             patch.dict("os.environ", {"ENVIRONMENT": "development"}, clear=False):
            from backend.tasks.scheduled_tasks import _do_daily_db_backup
            result = _do_daily_db_backup()

        assert result["skipped"] is True
        assert result["reason"] == "not_postgres"
        mock_alert.assert_not_called()

    @patch("backend.middleware.alerting.alert_backup_status")
    def test_not_postgres_production_alerts(self, mock_alert):
        """SQLite DB + ENVIRONMENT=production → alert fires (unexpected in prod)."""
        with patch("backend.tasks.scheduled_tasks.DATABASE_URL", "sqlite:///prod.db"), \
             patch.dict("os.environ", {"ENVIRONMENT": "production"}, clear=False):
            from backend.tasks.scheduled_tasks import _do_daily_db_backup
            result = _do_daily_db_backup()

        assert result["skipped"] is True
        mock_alert.assert_called_once()
        call_args = mock_alert.call_args
        assert "not PostgreSQL" in call_args[0][0]

    @patch("backend.middleware.alerting.alert_backup_status")
    def test_r2_not_configured_alerts(self, mock_alert):
        """Postgres DB but R2 unavailable → alert fires."""
        with patch("backend.tasks.scheduled_tasks.DATABASE_URL", "postgresql://u:p@host/db"), \
             patch("backend.services.storage_service.R2_AVAILABLE", False):
            from backend.tasks.scheduled_tasks import _do_daily_db_backup
            result = _do_daily_db_backup()

        assert result["skipped"] is True
        assert result["reason"] == "r2_not_configured"
        mock_alert.assert_called_once()
        assert "R2" in mock_alert.call_args[0][0]

    @patch("backend.middleware.alerting.alert_backup_status")
    def test_subprocess_failure_alerts(self, mock_alert):
        """pg_dump raises exception → alert fired with error detail."""
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = b"pg_dump: connection refused"

        with patch("backend.tasks.scheduled_tasks.DATABASE_URL", "postgresql://u:p@host/db"), \
             patch("backend.services.storage_service.R2_AVAILABLE", True), \
             patch("subprocess.run", return_value=mock_proc):
            from backend.tasks.scheduled_tasks import _do_daily_db_backup
            result = _do_daily_db_backup()

        assert result.get("success") is False
        mock_alert.assert_called_once()
        assert "Failed" in mock_alert.call_args[0][0]

    @patch("backend.middleware.alerting.alert_backup_status")
    def test_success_no_alert(self, mock_alert):
        """Successful backup → no alert (avoid daily noise)."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = b"fake dump data"

        with patch("backend.tasks.scheduled_tasks.DATABASE_URL", "postgresql://u:p@host/db"), \
             patch("backend.services.storage_service.R2_AVAILABLE", True), \
             patch("subprocess.run", return_value=mock_proc), \
             patch("backend.services.storage_service.store_pdf", return_value="backups/test.dump"):
            from backend.tasks.scheduled_tasks import _do_daily_db_backup
            result = _do_daily_db_backup()

        assert result.get("success") is True
        mock_alert.assert_not_called()


@patch("backend.middleware.alerting.SLACK_WEBHOOK_URL", "")
@patch("backend.middleware.alerting._send_slack")
def test_slack_webhook_unset_is_noop(mock_send_slack):
    """When SLACK_WEBHOOK_URL is empty, alert_backup_status is a safe no-op."""
    alert_backup_status("Failed", "Test detail")
    mock_send_slack.assert_not_called()
