"""Single-job gate with a small bounded waiting room."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.errors import ServiceBusyError


class JobLimiter:
    """Allow up to `concurrency` heavy jobs at a time and at most N waiting requests.

    ``asyncio.Semaphore(concurrency)`` is the actual gate; ``_waiting`` is used
    only for admission control so a flood of requests cannot queue up and consume RAM.
    """

    def __init__(self, max_waiting: int = 2, concurrency: int = 1) -> None:
        if max_waiting < 0:
            raise ValueError("max_waiting must be >= 0")
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self._max_waiting = max_waiting
        self._concurrency = concurrency
        self._waiting = 0
        self._active_count = 0
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(concurrency)

    @property
    def waiting(self) -> int:
        return self._waiting

    @property
    def active(self) -> bool:
        return self._active_count > 0

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Acquire a slot or raise ``ServiceBusyError`` (HTTP 429)."""

        is_waiter = False
        waiting_counted = False
        acquired = False

        async with self._lock:
            if self._active_count >= self._concurrency:
                if self._waiting >= self._max_waiting:
                    raise ServiceBusyError()
                self._waiting += 1
                is_waiter = True
                waiting_counted = True
            else:
                self._active_count += 1

        try:
            await self._semaphore.acquire()
            acquired = True

            if is_waiter:
                async with self._lock:
                    self._waiting -= 1
                    waiting_counted = False
                    self._active_count += 1

            yield

        finally:
            if waiting_counted:
                async with self._lock:
                    self._waiting -= 1
                    waiting_counted = False

            if acquired:
                async with self._lock:
                    self._active_count -= 1
                    self._semaphore.release()
