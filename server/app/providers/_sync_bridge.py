"""
Bridges a blocking/synchronous iterator (e.g. boto3's Bedrock event
stream, which has no async support at all) into an async generator, so
provider code can `async for` over it just like every other provider's
native async stream.

why a background thread instead of just calling next() in a loop: boto3's
EventStream blocks the calling thread while it waits for the next network
chunk. Calling that directly inside an `async def` would freeze the
*entire* event loop - including VAD/barge-in processing for every other
connected client - for the whole duration of the LLM response. Running
the blocking iteration in a dedicated thread and relaying items through a
thread-safe queue keeps the event loop free the whole time.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Iterable
from typing import Any

_SENTINEL = object()


async def bridge_sync_iterator(sync_iterable: Iterable[Any]) -> AsyncIterator[Any]:
    q: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            for item in sync_iterable:
                q.put(item)
        except Exception as exc:  # noqa: BLE001 - re-raised on the async side below
            q.put(exc)
        finally:
            q.put(_SENTINEL)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item
