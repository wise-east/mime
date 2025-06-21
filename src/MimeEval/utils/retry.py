import time
import logging
from functools import wraps
from typing import Callable, TypeVar, ParamSpec, Union, Type, Tuple
from google.api_core import exceptions as google_exceptions

logger = logging.getLogger(__name__)

P = ParamSpec('P')  # For arbitrary parameters
R = TypeVar('R', bound=str)  # Return type (string)

# Exceptions that should trigger a retry
RETRYABLE_EXCEPTIONS = (
    google_exceptions.ResourceExhausted,  # 429 quota errors
    google_exceptions.ServiceUnavailable,  # 503
    google_exceptions.DeadlineExceeded,   # timeout
    google_exceptions.InternalServerError, # 500
    ConnectionError,                      # network issues
    TimeoutError
)

def exponential_backoff(
    max_attempts: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 32.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = RETRYABLE_EXCEPTIONS
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator that implements exponential backoff for failed function calls.
    
    Args:
        max_attempts: Maximum number of attempts before giving up
        initial_delay: Initial delay between attempts in seconds
        max_delay: Maximum delay between attempts in seconds
        backoff_factor: Factor to multiply delay by after each attempt
        retryable_exceptions: Exception types that should trigger a retry
    
    Returns:
        Decorated function that will retry with exponential backoff
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        # On last attempt, re-raise the exception
                        raise
                    
                    # For quota errors, use a longer delay
                    if isinstance(e, google_exceptions.ResourceExhausted):
                        actual_delay = min(delay * 2, max_delay)  # More aggressive backoff for quota
                    else:
                        actual_delay = delay
                    
                    # Log the failure and prepare to retry
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}. "
                        f"Retrying in {actual_delay:.1f} seconds..."
                    )
                    
                    # Sleep with exponential backoff
                    time.sleep(actual_delay)
                    
                    # Increase delay for next attempt
                    delay = min(delay * backoff_factor, max_delay)
                except Exception as e:
                    # Non-retryable exception, raise immediately
                    logger.error(f"Non-retryable error occurred: {str(e)}")
                    raise
            
            # This should never be reached due to the raise in the loop
            raise last_exception
            
        return wrapper
    return decorator 