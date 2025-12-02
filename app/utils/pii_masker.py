"""
PII Masking Utility
Masks sensitive personal information in logs for GDPR/privacy compliance
"""

import re
from typing import Optional


def mask_email(email: Optional[str]) -> str:
    """
    Mask email address for logging.
    
    Examples:
        test@example.com → t***@example.com
        ab@domain.org → a*@domain.org
        a@b.com → a@b.com (too short to mask)
    """
    if not email or '@' not in email:
        return email or ""
    
    try:
        local, domain = email.split('@', 1)
        if len(local) <= 1:
            return email  # Can't mask single char
        elif len(local) == 2:
            return f"{local[0]}*@{domain}"
        else:
            return f"{local[0]}{'*' * (len(local) - 2)}{local[-1]}@{domain}"
    except Exception:
        return "***@***"


def mask_name(name: Optional[str]) -> str:
    """
    Mask person's name for logging.
    
    Examples:
        John Doe → J*** D**
        Alice → A****
        Jo → Jo (too short)
    """
    if not name:
        return ""
    
    try:
        parts = name.split()
        masked_parts = []
        for part in parts:
            if len(part) <= 2:
                masked_parts.append(part)
            else:
                masked_parts.append(f"{part[0]}{'*' * (len(part) - 1)}")
        return ' '.join(masked_parts)
    except Exception:
        return "***"


def mask_phone(phone: Optional[str]) -> str:
    """
    Mask phone number for logging.
    
    Examples:
        +1-555-123-4567 → +1-555-***-****
        1234567890 → ******7890
    """
    if not phone:
        return ""
    
    # Remove non-digits for processing
    digits = re.sub(r'\D', '', phone)
    
    if len(digits) <= 4:
        return phone  # Too short to mask
    
    # Keep last 4 digits visible
    return '*' * (len(digits) - 4) + digits[-4:]


def mask_ticket_text(text: Optional[str], max_length: int = 100) -> str:
    """
    Truncate and sanitize ticket text for logging.
    Removes potential PII patterns.
    
    Args:
        text: The ticket text to mask
        max_length: Maximum characters to show
    """
    if not text:
        return ""
    
    # Truncate
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    # Mask email patterns
    text = re.sub(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        '[EMAIL]',
        text
    )
    
    # Mask phone patterns (basic)
    text = re.sub(
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        '[PHONE]',
        text
    )
    
    return text


def mask_api_key(key: Optional[str]) -> str:
    """
    Mask API key for logging - show only first and last 4 chars.
    
    Examples:
        sk-abc123xyz789 → sk-a***789
    """
    if not key:
        return ""
    
    if len(key) <= 8:
        return '*' * len(key)
    
    return f"{key[:4]}***{key[-4:]}"


def create_safe_log_context(
    ticket_id: Optional[int] = None,
    email: Optional[str] = None,
    name: Optional[str] = None,
    subject: Optional[str] = None,
) -> dict:
    """
    Create a safe logging context with masked PII.
    
    Returns a dict suitable for structured logging.
    """
    context = {}
    
    if ticket_id is not None:
        context["ticket_id"] = ticket_id
    
    if email:
        context["requester_email"] = mask_email(email)
    
    if name:
        context["requester_name"] = mask_name(name)
    
    if subject:
        context["subject"] = mask_ticket_text(subject, max_length=50)
    
    return context
