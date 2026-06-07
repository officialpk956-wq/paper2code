import requests
import os

BASE_URL = "http://127.0.0.1:8000"

def test_size(size_mb, filename):
    with open(filename, "wb") as f:
        f.write(b"0" * (size_mb * 1024 * 1024))
    
    with open(filename, "rb") as f:
        files = {"file": (filename, f, "application/pdf")}
        res = requests.post(f"{BASE_URL}/api/papers/upload", files=files)
        print(f"[{size_mb}MB] Status: {res.status_code}, Response: {res.text}")

    os.remove(filename)

print("Testing PDF Sizes...")
test_size(5, "test_5mb.pdf")
test_size(10, "test_10mb.pdf")
test_size(20, "test_20mb.pdf")
test_size(21, "test_21mb.pdf")
