"""Single-job gate with a small bounded waiting room."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.errors import ServiceBusyError


class JobLimiter:
    """Allow one heavy job at a time and at most N waiting requests.

    ``asyncio.Semaphore(1)`` is the actual gate; ``_waiting`` is used only for
    admission control so a flood of requests cannot queue up and consume RAM.
    """

    def __init__(self, max_waiting: int = 2) -> None:
        if max_waiting < 0:
            raise ValueError("max_waiting must be >= 0")
        self._max_waiting = max_waiting
        self._waiting = 0
        self._active = False
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(1)

    @property
    def waiting(self) -> int:
        return self._waiting

    @property
    def active(self) -> bool:
        return self._active

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Acquire a slot or raise ``ServiceBusyError`` (HTTP 429)."""

        is_waiter = False
        waiting_counted = False
        acquired = False

        async with self._lock:
            if self._active:
                if self._waiting >= self._max_waiting:
                    raise ServiceBusyError()
                self._waiting += 1
                is_waiter = True
                waiting_counted = True
            else:
                self._active = True

        try:
            await self._semaphore.acquire()
            acquired = True

            if is_waiter:
                async with self._lock:
                    self._waiting -= 1
                    waiting_counted = False

            yield

        finally:
            # Cancellation can happen between acquiring the permit and
            # finishing the bookkeeping, so make cleanup idempotent.
            if waiting_counted:
                async with self._lock:
                    self._waiting -= 1
                    waiting_counted = False

            if acquired:
                await self._release()

    async def _release(self) -> None:
        async with self._lock:
            if self._waiting > 0:
                # Hand the permit directly to a waiter; active stays true.
                self._semaphore.release()
            else:
                self._active = False
                self._semaphore.release()
