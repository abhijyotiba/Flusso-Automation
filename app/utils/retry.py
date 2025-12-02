"""
Retry Logic Utilities
Provides retry decorators for external API calls with exponential backoff
"""

import logging
from functools import wraps
from typing import Callable, Type, Tuple, Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
)
import requests

logger = logging.getLogger(__name__)


# Common transient exceptions that should trigger retries
TRANSIENT_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    ConnectionResetError,
    TimeoutError,
)


def create_retry_decorator(
    max_attempts: int = 3,
    min_wait: float = 1,
    max_wait: float = 10,
    exceptions: Tuple[Type[Exception], ...] = TRANSIENT_EXCEPTIONS,
):
    """
    Create a retry decorator with configurable parameters.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        exceptions: Tuple of exception types to retry on
    
    Returns:
        A tenacity retry decorator
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.DEBUG),
        reraise=True,
    )


# Pre-configured decorators for different services
retry_api_call = create_retry_decorator(
    max_attempts=3,
    min_wait=1,
    max_wait=10,
)

retry_external_service = create_retry_decorator(
    max_attempts=3,
    min_wait=2,
    max_wait=30,
)

retry_embedding = create_retry_decorator(
    max_attempts=2,
    min_wait=1,
    max_wait=5,
)


def with_retry(
    max_attempts: int = 3,
    exceptions: Tuple[Type[Exception], ...] = TRANSIENT_EXCEPTIONS,
    on_retry: Callable[[Exception, int], None] = None,
):
    """
    Decorator factory for adding retry logic to functions.
    
    Args:
        max_attempts: Maximum retry attempts
        exceptions: Exception types to catch
        on_retry: Optional callback when retry occurs (exception, attempt_number)
    
    Usage:
        @with_retry(max_attempts=3)
        def my_api_call():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        wait_time = min(2 ** attempt, 10)  # Exponential backoff
                        logger.warning(
                            f"[RETRY] {func.__name__} failed (attempt {attempt}/{max_attempts}): {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        if on_retry:
                            on_retry(e, attempt)
                        import time
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"[RETRY] {func.__name__} failed after {max_attempts} attempts: {e}"
                        )
            raise last_exception
        return wrapper
    return decorator
