"""
Async I/O utilities for non-blocking operations.

Provides async wrappers for CPU-intensive and I/O-bound operations
to keep the UI responsive during heavy computations.
"""

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

# Global thread pool for CPU-bound operations
_executor = ThreadPoolExecutor(max_workers=4)

T = TypeVar("T")


def run_in_thread(func: Callable[..., T]) -> Callable[..., asyncio.Future[T]]:
    """
    Decorator to run a synchronous function in a thread pool.

    Usage:
        @run_in_thread
        def heavy_computation(data):
            return expensive_operation(data)

        result = await heavy_computation(data)
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, functools.partial(func, *args, **kwargs))

    return wrapper


def shutdown_executor():
    """Shutdown the global thread pool executor. Call on app exit."""
    _executor.shutdown(wait=True)
