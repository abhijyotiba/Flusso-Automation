"""
Audit Logging Utility
Adds structured audit events into TicketState.audit_events
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Maximum audit events to prevent unbounded memory growth
MAX_AUDIT_EVENTS = 500


def add_audit_event(
    state: Dict[str, Any],
    event: str,
    event_type: str = "INFO",
    details: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Append a structured audit event to the state's audit_events list.
    
    Includes size limit protection to prevent memory issues.

    Args:
        state: Current state object
        event: Name of the event (e.g., "fetch_ticket", "vision_rag")
        event_type: INFO | ERROR | SUCCESS | FETCH | DECISION
        details: Additional info dictionary

    Returns:
        Updated state partial: {"audit_events": [...]}
    """

    details = details or {}

    audit_events: List[Dict[str, Any]] = state.get("audit_events", []) or []

    audit_events.append({
        "event": event,
        "type": event_type,
        "details": details
    })

    # Prevent unbounded growth - keep only most recent events
    if len(audit_events) > MAX_AUDIT_EVENTS:
        removed_count = len(audit_events) - MAX_AUDIT_EVENTS
        audit_events = audit_events[-MAX_AUDIT_EVENTS:]
        logger.warning(f"[AUDIT] Truncated {removed_count} old events (limit: {MAX_AUDIT_EVENTS})")

    return {"audit_events": audit_events}
