import redis
try:
    r = redis.from_url("rediss://localhost:6379/0", ssl_cert_reqs="CERT_NONE")
    print("Success with kwargs!")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")
