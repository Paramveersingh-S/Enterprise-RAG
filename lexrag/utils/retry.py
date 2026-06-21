import functools
from typing import Any, Callable, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lexrag.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=Callable[..., Any])

def with_exponential_backoff(
    exceptions: tuple[type[Exception], ...] = (Exception,),
    max_retries: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 8.0,
) -> Callable[[T], T]:
    """Decorator to apply exponential backoff to a function.
    
    Args:
        exceptions: Tuple of exceptions to catch and retry on.
        max_retries: Maximum number of retry attempts.
        min_wait: Minimum wait time in seconds.
        max_wait: Maximum wait time in seconds.
        
    Returns:
        Decorated function.
    """
    def before_retry_log(retry_state: Any) -> None:
        logger.warning(
            f"Retrying after error",
            attempt=retry_state.attempt_number,
            exception=str(retry_state.outcome.exception()),
        )

    return retry(
        retry=retry_if_exception_type(exceptions),
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        before_sleep=before_retry_log,
    ) # type: ignore
