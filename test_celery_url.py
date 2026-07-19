from celery.backends.redis import RedisBackend
from celery import Celery
import ssl

print("Testing celery redis parsing")
app = Celery("test", broker="rediss://localhost:6379/0", backend="rediss://localhost:6379/0")
_ssl_opts = {"ssl_cert_reqs": ssl.CERT_NONE}
app.conf.broker_use_ssl = _ssl_opts
app.conf.redis_backend_use_ssl = _ssl_opts

try:
    print(app.backend)
except Exception as e:
    print(f"Backend Exception: {e}")
