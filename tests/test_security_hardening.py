import io
import unittest.mock
import pytest
from fastapi.testclient import TestClient

from app.config import (
    MAX_DATASET_COLUMNS,
    MAX_DATASET_ROWS,
    MAX_UPLOAD_SIZE_BYTES,
)
from app.data.explorer import DatasetExplorerError, read_dataset_csv
from app.main import app
from app.monitoring.webhooks import is_safe_webhook_url, deliver_webhook

client = TestClient(app)


# -------------------------------------------------------------------
# 1. CSV Security & Upload Boundaries
# -------------------------------------------------------------------

def test_csv_upload_exceeding_size_limit():
    """Upload exceeding MAX_UPLOAD_SIZE_BYTES should be rejected with 413 or error."""
    oversized = b"a,b\n" + (b"1,2\n" * ((MAX_UPLOAD_SIZE_BYTES // 4) + 10))
    assert len(oversized) > MAX_UPLOAD_SIZE_BYTES

    response = client.post(
        "/api/v1/drift/upload",
        files={
            "baseline": ("base.csv", io.BytesIO(oversized), "text/csv"),
            "current": ("curr.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv"),
        },
    )
    assert response.status_code == 413
    assert "limit" in response.json()["detail"].lower()


def test_csv_upload_empty_file():
    """Empty CSV files must be rejected safely."""
    response = client.post(
        "/api/v1/drift/upload",
        files={
            "baseline": ("base.csv", io.BytesIO(b""), "text/csv"),
            "current": ("curr.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv"),
        },
    )
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


def test_csv_raw_duplicate_headers_detected_not_mangled():
    """Pandas would rename 'score,score' to 'score,score.1'. Security check must catch raw duplicate."""
    duplicate_header_csv = b"age,score,score\n20,80,90\n"

    # In Dataset Explorer
    with pytest.raises(DatasetExplorerError) as exc_info:
        read_dataset_csv("test.csv", duplicate_header_csv)
    assert "duplicate" in str(exc_info.value).lower()

    # In drift upload endpoint
    response = client.post(
        "/api/v1/drift/upload",
        files={
            "baseline": ("base.csv", io.BytesIO(duplicate_header_csv), "text/csv"),
            "current": ("curr.csv", io.BytesIO(b"age,score\n20,80\n"), "text/csv"),
        },
    )
    assert response.status_code == 422
    assert "duplicate" in response.json()["detail"].lower()


def test_csv_empty_column_name_rejected():
    """CSV with empty header name (e.g. 'age,,score') must be rejected."""
    empty_col_csv = b"age,,score\n20,1,80\n"

    with pytest.raises(DatasetExplorerError):
        read_dataset_csv("test.csv", empty_col_csv)

    response = client.post(
        "/api/v1/drift/upload",
        files={
            "baseline": ("base.csv", io.BytesIO(empty_col_csv), "text/csv"),
            "current": ("curr.csv", io.BytesIO(b"age,score\n20,80\n"), "text/csv"),
        },
    )
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


def test_csv_exceeding_column_limit():
    """CSV with columns exceeding MAX_DATASET_COLUMNS must be rejected."""
    cols = [f"c{i}" for i in range(MAX_DATASET_COLUMNS + 5)]
    header = ",".join(cols).encode("utf-8") + b"\n"
    row = ",".join(["1"] * len(cols)).encode("utf-8") + b"\n"
    content = header + row

    with pytest.raises(DatasetExplorerError) as exc_info:
        read_dataset_csv("test.csv", content)
    assert "column" in str(exc_info.value).lower()


def test_csv_null_bytes_rejected():
    """Files with embedded null bytes must be rejected safely."""
    poisoned = b"a,b\x00,c\n1,2,3\n"
    with pytest.raises(DatasetExplorerError):
        read_dataset_csv("test.csv", poisoned)


# -------------------------------------------------------------------
# 2. SSRF Webhook Destination Protections
# -------------------------------------------------------------------

def test_webhook_ssrf_rejects_loopback():
    """Loopback addresses must be blocked from webhook registration."""
    is_safe, reason = is_safe_webhook_url("http://127.0.0.1:8000/webhook")
    assert is_safe is False
    assert "restricted" in reason.lower() or "not permitted" in reason.lower()


def test_webhook_ssrf_rejects_localhost_hostname():
    """Localhost hostname must be blocked."""
    is_safe, reason = is_safe_webhook_url("http://localhost:9000/hook")
    assert is_safe is False
    assert "not permitted" in reason.lower()


def test_webhook_ssrf_rejects_cloud_metadata():
    """AWS/GCP/Azure link-local metadata address (169.254.169.254) must be blocked."""
    is_safe, reason = is_safe_webhook_url("http://169.254.169.254/latest/meta-data/")
    assert is_safe is False
    assert "restricted" in reason.lower()


def test_webhook_ssrf_rejects_private_class_a_b_c():
    """Private IP ranges (10.x, 172.16.x, 192.168.x) must be blocked."""
    for private_url in [
        "http://10.0.0.1:8080/hook",
        "http://172.16.0.5:8000/hook",
        "http://192.168.1.1:3000/hook",
    ]:
        is_safe, _ = is_safe_webhook_url(private_url)
        assert is_safe is False, f"Failed to reject private URL: {private_url}"


def test_webhook_ssrf_rejects_embedded_credentials():
    """URLs with user:password embedded should be rejected."""
    is_safe, reason = is_safe_webhook_url("https://admin:secret@example.com/webhook")
    assert is_safe is False
    assert "credential" in reason.lower()


def test_webhook_ssrf_rejects_non_http_schemes():
    """Non-HTTP schemes (ftp, file, gopher) must be rejected."""
    for scheme_url in ["file:///etc/passwd", "gopher://evil.com", "ftp://evil.com/hook"]:
        is_safe, _ = is_safe_webhook_url(scheme_url)
        assert is_safe is False


def test_webhook_deliver_blocks_unsafe_url_without_outbound_network():
    """deliver_webhook should immediately abort without making an outbound socket call."""
    success, code, msg = deliver_webhook("http://127.0.0.1:8080/evil", {"event": "test"})
    assert success is False
    assert "blocked unsafe webhook" in msg.lower()


def test_webhook_test_endpoint_rejects_ssrf():
    """The /api/v1/webhooks/test endpoint must reject SSRF targets with HTTP 400."""
    response = client.post(
        "/api/v1/webhooks/test",
        json={"url": "http://127.0.0.1:8000/internal", "name": "Evil", "service_type": "generic"},
    )
    # Returns 400 or a JSON payload with success=False
    if response.status_code == 200:
        data = response.json()
        assert data["success"] is False
        assert "not permitted" in data["message"].lower()
    else:
        assert response.status_code in (400, 422)


# -------------------------------------------------------------------
# 3. Safe Global Error Handling
# -------------------------------------------------------------------

def test_global_exception_handler_returns_safe_500():
    """Unhandled exceptions must return uniform error messages without leaking tracebacks."""
    with unittest.mock.patch("app.main.list_webhooks", side_effect=RuntimeError("Database /var/run/secret failed")):
        response = client.get("/api/v1/webhooks")
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        # Traceback and internal paths must NOT be in client response
        assert "var/run/secret" not in data["detail"]
        assert "RuntimeError" not in data["detail"]


# -------------------------------------------------------------------
# 4. API Key Authentication (Configurable)
# -------------------------------------------------------------------

def test_api_key_auth_enforced_when_configured():
    """When MODELAUTOPSY_API_KEY is configured, mutating webhook endpoints require valid X-API-Key."""
    with unittest.mock.patch("app.main.MODELAUTOPSY_API_KEY", "super-secret-production-token"):
        # Without key -> 401
        res_no_key = client.post(
            "/api/v1/webhooks",
            json={
                "name": "Secured Hook",
                "url": "https://hooks.slack.com/services/T1/B1/X1",
                "service_type": "slack",
                "min_severity": "high",
            },
        )
        assert res_no_key.status_code == 401
        assert "api key" in res_no_key.json()["detail"].lower()

        # With invalid key -> 401
        res_bad_key = client.post(
            "/api/v1/webhooks",
            headers={"X-API-Key": "wrong-key"},
            json={
                "name": "Secured Hook",
                "url": "https://hooks.slack.com/services/T1/B1/X1",
                "service_type": "slack",
                "min_severity": "high",
            },
        )
        assert res_bad_key.status_code == 401

        # With correct key -> passes auth (either 200 or downstream validation)
        res_good_key = client.post(
            "/api/v1/webhooks",
            headers={"X-API-Key": "super-secret-production-token"},
            json={
                "name": "Secured Hook",
                "url": "https://hooks.slack.com/services/T1/B1/X1",
                "service_type": "slack",
                "min_severity": "high",
            },
        )
        assert res_good_key.status_code != 401
