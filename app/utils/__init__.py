"""Utility functions and helpers"""

from app.utils.audit import add_audit_event
from app.utils.pii_masker import mask_email, mask_name, mask_phone, mask_ticket_text, mask_api_key
from app.utils.retry import retry_api_call, retry_external_service, with_retry
from app.utils.validation import requires_fields, requires_any_field, validate_state_type, NodeValidationError

__all__ = [
    # Audit
    "add_audit_event",
    # PII Masking
    "mask_email",
    "mask_name",
    "mask_phone",
    "mask_ticket_text",
    "mask_api_key",
    # Retry Logic
    "retry_api_call",
    "retry_external_service",
    "with_retry",
    # Validation
    "requires_fields",
    "requires_any_field",
    "validate_state_type",
    "NodeValidationError",
]
