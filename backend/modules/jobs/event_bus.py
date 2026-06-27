import uuid
import datetime
from typing import Dict, Any, List, Callable

class Event:
    def __init__(self, name: str, version: str, data: Dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.name = name
        self.version = version
        self.data = data
        self.timestamp = datetime.datetime.utcnow()

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}

    def subscribe(self, event_name: str, handler: Callable[[Event], None]) -> None:
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(handler)

    def publish(self, event_name: str, version: str, data: Dict[str, Any]) -> None:
        event = Event(event_name, version, data)
        if event_name in self._subscribers:
            for handler in self._subscribers[event_name]:
                try:
                    # In production event bus, handler runs asynchronously or via worker queue.
                    # We run it synchronously or dispatch to background task runner.
                    handler(event)
                except Exception:
                    pass
