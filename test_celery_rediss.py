import os
os.environ["REDIS_URL"] = "rediss://localhost:6379/0"
from backend.celery_app import celery_app
print(celery_app.conf.broker_url)

try:
    with celery_app.pool.acquire(block=True) as conn:
        print("Kombu Broker URL:", conn.as_uri())
except Exception as e:
    print(f"Exception connecting to broker: {e}")
