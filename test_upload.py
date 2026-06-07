import io
from fastapi.testclient import TestClient
from backend.server import app

client = TestClient(app)

def test_upload():
    # 1. Test non-PDF
    res = client.post("/api/parse_pdf", files={"file": ("test.txt", b"hello", "text/plain")})
    assert res.status_code == 400
    assert "Only PDF files are supported" in res.json()["detail"]
    print("Non-PDF test passed")

    # 2. Test 21MB PDF
    large_pdf = b"0" * (21 * 1024 * 1024)
    res = client.post("/api/parse_pdf", files={"file": ("large.pdf", large_pdf, "application/pdf")})
    assert res.status_code == 400
    assert "File exceeds 20MB limit" in res.json()["detail"]
    print("21MB PDF test passed")

if __name__ == "__main__":
    test_upload()
    print("All tests passed.")
