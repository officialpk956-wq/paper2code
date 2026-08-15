"""
tests/test_rate_limit_zset_collision.py

Tests for Redis ZSET rate limiting:
1. Frozen-time burst test: 10 requests at the exact same timestamp with limit=5.
   First 5 requests succeed, 6th through 10th are rejected (prevents second-resolution collision bypass).
2. Normal sliding window progression across time.
3. Old entries outside the sliding window are expired via zremrangebyscore.
4. In-memory fallback branch when Redis is unavailable.
"""

import time
import pytest
from unittest.mock import patch

from backend.modules.auth.middleware.rate_limit import (
    check_rate_limit,
    _in_memory_store,
)


class MockRedisPipeline:
    def __init__(self, parent):
        self.parent = parent
        self.commands = []

    def zremrangebyscore(self, key, min_s, max_s):
        self.commands.append(("zremrangebyscore", key, min_s, max_s))
        return self

    def zadd(self, key, mapping):
        self.commands.append(("zadd", key, mapping))
        return self

    def zcard(self, key):
        self.commands.append(("zcard", key))
        return self

    def expire(self, key, seconds):
        self.commands.append(("expire", key, seconds))
        return self

    def execute(self):
        results = []
        for cmd in self.commands:
            op = cmd[0]
            if op == "zremrangebyscore":
                _, key, min_s, max_s = cmd
                zset = self.parent.zsets.setdefault(key, {})
                to_remove = [m for m, s in zset.items() if min_s <= s <= max_s]
                for m in to_remove:
                    del zset[m]
                results.append(len(to_remove))
            elif op == "zadd":
                _, key, mapping = cmd
                zset = self.parent.zsets.setdefault(key, {})
                added = 0
                for m, s in mapping.items():
                    if m not in zset:
                        added += 1
                    zset[m] = s  # ZADD updates score if exists, adds if not
                results.append(added)
            elif op == "zcard":
                _, key = cmd
                zset = self.parent.zsets.setdefault(key, {})
                results.append(len(zset))
            elif op == "expire":
                results.append(True)
        return results


class MockRedisClient:
    def __init__(self):
        self.zsets = {}

    def pipeline(self):
        return MockRedisPipeline(self)


def test_frozen_time_burst_rate_limiting():
    """
    Core regression test:
    Simulate a sub-second burst of 10 requests at the EXACT same timestamp (frozen time.time()).
    With limit=5, requests 1-5 must return True, and requests 6-10 must return False.
    (On the buggy code, all 10 would return True due to member collision str(now)).
    """
    mock_redis = MockRedisClient()
    frozen_timestamp = 1700000000.123456

    with patch("backend.modules.auth.middleware.rate_limit._redis_client", mock_redis), \
         patch("time.time", return_value=frozen_timestamp):

        results = []
        for _ in range(10):
            res = check_rate_limit("user:burst_test", limit=5, window_seconds=60)
            results.append(res)

        # First 5 succeed
        assert results[:5] == [True, True, True, True, True]
        # 6th through 10th fail
        assert results[5:] == [False, False, False, False, False]

        # Verify Redis ZSET contains unique members with correct score
        zset = mock_redis.zsets["user:burst_test"]
        assert len(zset) == 10
        for member, score in zset.items():
            assert score == frozen_timestamp
            assert member.startswith(str(frozen_timestamp) + ":")


def test_sliding_window_progression_and_expiry():
    """Test that requests spread across time work, and old entries are purged correctly."""
    mock_redis = MockRedisClient()
    base_time = 1700000000.0

    with patch("backend.modules.auth.middleware.rate_limit._redis_client", mock_redis):
        # Time T=0: 3 requests
        with patch("time.time", return_value=base_time):
            assert check_rate_limit("user:progression", limit=5, window_seconds=10) is True
            assert check_rate_limit("user:progression", limit=5, window_seconds=10) is True
            assert check_rate_limit("user:progression", limit=5, window_seconds=10) is True

        # Time T=5s: 2 more requests (total 5 in window) -> should pass
        with patch("time.time", return_value=base_time + 5):
            assert check_rate_limit("user:progression", limit=5, window_seconds=10) is True
            assert check_rate_limit("user:progression", limit=5, window_seconds=10) is True
            # 6th request at T=5s should fail
            assert check_rate_limit("user:progression", limit=5, window_seconds=10) is False

        # Time T=12s: first 3 requests from T=0 have expired (outside 10s window)
        # Remaining in window: 3 requests from T=5s. Adding 2 requests should succeed.
        with patch("time.time", return_value=base_time + 12):
            assert check_rate_limit("user:progression", limit=5, window_seconds=10) is True
            assert check_rate_limit("user:progression", limit=5, window_seconds=10) is True
            # Reached limit (5) again
            assert check_rate_limit("user:progression", limit=5, window_seconds=10) is False


def test_in_memory_fallback_burst_and_window():
    """Test that the in-memory fallback behaves properly when Redis is down."""
    _in_memory_store.clear()
    frozen_timestamp = 1700000000.0

    with patch("backend.modules.auth.middleware.rate_limit._redis_client", None), \
         patch("time.time", return_value=frozen_timestamp):

        # Burst of 5 requests
        for _ in range(5):
            assert check_rate_limit("in_mem_key", limit=5, window_seconds=60) is True

        # 6th request is rejected
        assert check_rate_limit("in_mem_key", limit=5, window_seconds=60) is False

    # Window expiry in-memory
    with patch("backend.modules.auth.middleware.rate_limit._redis_client", None), \
         patch("time.time", return_value=frozen_timestamp + 65):
        # Old entries purged, request succeeds
        assert check_rate_limit("in_mem_key", limit=5, window_seconds=60) is True
