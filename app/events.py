"""
Lightweight in-process event bus for SSE broadcasting.

A set of asyncio.Queue objects — one per connected SSE client — is maintained.
Any synchronous code (route handlers, background tasks) calls ``publish`` to
push an event dict to all live queues.  The SSE endpoint drains each client's
queue and formats the data as SSE text frames.
"""

from __future__ import annotations

import asyncio
import json
import weakref
from typing import Any


# Use weak references so that queues are garbage-collected when their
# consumer coroutine finishes without needing explicit cleanup on error paths.
_clients: weakref.WeakSet[asyncio.Queue] = weakref.WeakSet()  # type: ignore[type-arg]
_lock = asyncio.Lock()


def publish(event_type: str, data: dict[str, Any] | None = None) -> None:
    """
    Broadcast an event to every connected SSE client.

    Safe to call from synchronous code; schedules the push onto each
    client's queue using the running event loop when one is available.
    """
    payload = {"type": event_type, **(data or {})}
    encoded = json.dumps(payload, allow_nan=False)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return  # No event loop — server hasn't fully started yet

    # Snapshot to avoid mutation during iteration
    for queue in list(_clients):
        try:
            loop.call_soon_threadsafe(queue.put_nowait, encoded)
        except asyncio.QueueFull:
            pass  # Slow client — drop event rather than block


async def subscribe() -> asyncio.Queue:
    """Create and register a new client queue."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    async with _lock:
        _clients.add(queue)
    return queue


async def unsubscribe(queue: asyncio.Queue) -> None:
    """Remove a client queue (called when the SSE connection closes)."""
    async with _lock:
        _clients.discard(queue)
