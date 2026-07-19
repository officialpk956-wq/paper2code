import ssl
import redis

# Test with integer
try:
    redis.from_url("rediss://localhost", ssl_cert_reqs=ssl.CERT_NONE)
    print("Int passed")
except Exception as e:
    print(f"Int failed: {e}")

# Test with string
try:
    redis.from_url("rediss://localhost", ssl_cert_reqs="CERT_NONE")
    print("String passed")
except Exception as e:
    print(f"String failed: {e}")
