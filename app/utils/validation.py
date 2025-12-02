"""
Input Validation Utilities
Provides decorators for validating node inputs
"""

import logging
from functools import wraps
from typing import Callable, Any, List

logger = logging.getLogger(__name__)


class NodeValidationError(Exception):
    """Raised when a node receives invalid input state."""
    pass


def requires_fields(*fields: str):
    """
    Decorator to validate that required fields exist in state before node execution.
    
    Args:
        *fields: Field names that must exist and be non-None in state
    
    Raises:
        NodeValidationError: If any required field is missing
    
    Usage:
        @requires_fields("ticket_id", "ticket_text")
        def my_node(state: TicketState) -> dict:
            # Safe to access ticket_id and ticket_text
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: dict) -> Any:
            missing = []
            for field in fields:
                value = state.get(field)
                if value is None:
                    missing.append(field)
                elif isinstance(value, str) and not value.strip():
                    missing.append(f"{field} (empty)")
            
            if missing:
                error_msg = f"{func.__name__} requires fields: {missing}"
                logger.error(f"[VALIDATION] {error_msg}")
                raise NodeValidationError(error_msg)
            
            return func(state)
        return wrapper
    return decorator


def requires_any_field(*fields: str):
    """
    Decorator to validate that at least one of the specified fields exists.
    
    Args:
        *fields: At least one of these fields must exist and be non-None
    
    Raises:
        NodeValidationError: If none of the fields exist
    
    Usage:
        @requires_any_field("ticket_text", "ticket_images")
        def my_node(state: TicketState) -> dict:
            # At least one of ticket_text or ticket_images exists
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: dict) -> Any:
            has_any = False
            for field in fields:
                value = state.get(field)
                if value is not None:
                    if isinstance(value, (list, dict, str)):
                        if len(value) > 0:
                            has_any = True
                            break
                    else:
                        has_any = True
                        break
            
            if not has_any:
                error_msg = f"{func.__name__} requires at least one of: {fields}"
                logger.error(f"[VALIDATION] {error_msg}")
                raise NodeValidationError(error_msg)
            
            return func(state)
        return wrapper
    return decorator


def validate_state_type(field: str, expected_type: type):
    """
    Decorator to validate that a field has the expected type.
    
    Args:
        field: The field name to check
        expected_type: The expected type (e.g., str, int, list)
    
    Usage:
        @validate_state_type("ticket_id", int)
        def my_node(state: TicketState) -> dict:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: dict) -> Any:
            value = state.get(field)
            if value is not None and not isinstance(value, expected_type):
                error_msg = f"{func.__name__}: {field} must be {expected_type.__name__}, got {type(value).__name__}"
                logger.error(f"[VALIDATION] {error_msg}")
                raise NodeValidationError(error_msg)
            
            return func(state)
        return wrapper
    return decorator
