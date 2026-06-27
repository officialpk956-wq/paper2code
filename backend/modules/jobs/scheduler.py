import time
import threading
import datetime
from typing import Dict, Any, Callable, List

class ScheduledTask:
    def __init__(self, name: str, interval_seconds: int, action: Callable[[], None], run_once: bool = False):
        self.name = name
        self.interval_seconds = interval_seconds
        self.action = action
        self.run_once = run_once
        self.last_run: Optional[datetime.datetime] = None
        self.next_run = datetime.datetime.utcnow()

class Scheduler:
    def __init__(self):
        self.tasks: List[ScheduledTask] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def schedule_interval(self, name: str, interval_seconds: int, action: Callable[[], None], run_once: bool = False) -> None:
        task = ScheduledTask(name, interval_seconds, action, run_once)
        self.tasks.append(task)

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread:
            self._stop_event.set()
            self._thread.join(timeout=2.0)

    def trigger_run_now(self) -> None:
        """Helper to run pending tasks immediately for testing/validation."""
        now = datetime.datetime.utcnow()
        for task in list(self.tasks):
            if now >= task.next_run:
                try:
                    task.action()
                except Exception:
                    pass
                task.last_run = now
                if task.run_once:
                    self.tasks.remove(task)
                else:
                    task.next_run = now + datetime.timedelta(seconds=task.interval_seconds)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self.trigger_run_now()
            time.sleep(0.5)
