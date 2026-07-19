from celery.backends.redis import RedisBackend
from celery import Celery

_orig_init = RedisBackend.__init__
def patched_init(self, *args, **kwargs):
    print(f"args: {args}, kwargs: {kwargs}")
    try:
        _orig_init(self, *args, **kwargs)
    except Exception as e:
        print(f"connparams at crash: {getattr(self, 'connparams', None)}")
        raise e
RedisBackend.__init__ = patched_init

try:
    app2 = Celery("test_app2", backend="rediss://localhost:6379/0?ssl_cert_reqs=CERT_NONE")
    b2 = app2.backend
except Exception as e:
    print(f"FAILED2: {type(e).__name__}: {e}")
