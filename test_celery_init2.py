from celery.backends.redis import RedisBackend
from celery import Celery
import traceback

try:
    app2 = Celery("test_app2", backend="rediss://localhost:6379/0?ssl_cert_reqs=CERT_NONE")
    b2 = app2.backend
except Exception as e:
    print("FAILED2:")
    traceback.print_exc()
