from celery import Celery
import os
os.environ["CELERY_RESULT_BACKEND"] = "rediss://localhost:6379/0"

app = Celery("test_app", backend="rediss://localhost:6379/0")
print("App created")
try:
    b = app.backend
    print("Backend created")
except Exception as e:
    print(f"FAILED: {e}")

app2 = Celery("test_app2", backend="rediss://localhost:6379/0?ssl_cert_reqs=CERT_NONE")
try:
    b2 = app2.backend
    print("Backend2 created")
except Exception as e:
    print(f"FAILED2: {e}")
