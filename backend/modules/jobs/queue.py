import json
import logging
import os
from typing import Any

import redis

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class QueueBackend:
    def enqueue(self, queue_name: str, job_id: str, payload: dict[str, Any]) -> None:
        raise NotImplementedError()

    def dequeue(self, queue_name: str) -> tuple[str, dict[str, Any]] | None:
        raise NotImplementedError()


class RedisQueue(QueueBackend):
    def __init__(self):
        try:
            self.client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            self.client.ping()
        except Exception as e:
            logger.warning(f"Redis queue backend offline: {e}")
            self.client = None

        # In-memory queue fallback
        self._in_memory_queues = {}

    def _get_queue_key(self, name: str) -> str:
        return f"queue:{name}"

    def enqueue(self, queue_name: str, job_id: str, payload: dict[str, Any]) -> None:
        data = json.dumps({"job_id": job_id, "payload": payload})
        if self.client:
            try:
                self.client.rpush(self._get_queue_key(queue_name), data)
                return
            except Exception as e:
                logger.exception(f"Redis enqueue error: {e}")

        # In-memory fallback
        if queue_name not in self._in_memory_queues:
            self._in_memory_queues[queue_name] = []
        self._in_memory_queues[queue_name].append(data)

    def dequeue(self, queue_name: str) -> tuple[str, dict[str, Any]] | None:
        if self.client:
            try:
                data = self.client.lpop(self._get_queue_key(queue_name))
                if data:
                    item = json.loads(data)
                    return item["job_id"], item["payload"]
            except Exception as e:
                logger.exception(f"Redis dequeue error: {e}")

        # In-memory fallback
        if queue_name in self._in_memory_queues and self._in_memory_queues[queue_name]:
            data = self._in_memory_queues[queue_name].pop(0)
            item = json.loads(data)
            return item["job_id"], item["payload"]
        return None
