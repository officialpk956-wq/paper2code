import redis
try:
    r = redis.from_url("rediss://localhost:6379/0")
    print("Success")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")
