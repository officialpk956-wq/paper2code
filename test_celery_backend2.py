from celery import Celery
import os

if "CELERY_RESULT_BACKEND" in os.environ:
    del os.environ["CELERY_RESULT_BACKEND"]

app2 = Celery("test_app2", backend="rediss://localhost:6379/0?ssl_cert_reqs=CERT_NONE")
try:
    b2 = app2.backend
    print("Backend2 created successfully!")
except Exception as e:
    print(f"FAILED2: {e}")
