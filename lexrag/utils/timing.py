import asyncio
import time
from contextlib import contextmanager
from typing import Generator, Any

from lexrag.logger import get_logger

logger = get_logger(__name__)

@contextmanager
def timing_context(operation_name: str) -> Generator[None, None, None]:
    """Context manager to measure the latency of a block of code.
    
    Args:
        operation_name: The name of the operation being measured.
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        logger.info(f"{operation_name} completed", latency_ms=latency_ms)

class AsyncTimer:
    """Async context manager to measure the latency of an async block of code."""
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = 0.0

    async def __aenter__(self) -> "AsyncTimer":
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        end_time = time.perf_counter()
        latency_ms = (end_time - self.start_time) * 1000
        logger.info(f"{self.operation_name} completed", latency_ms=latency_ms)
