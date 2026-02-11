"""
Workflow Log Builder
Transforms TicketState into centralized log format.
"""

import logging
import os
from typing import Dict, Any
from datetime import datetime

from app.graph.state import TicketState
from app.utils.workflow_log_schema import (
    WorkflowLogSchema,
    hash_pii,
    sanitize_trace,
    to_json_safe
)

logger = logging.getLogger(__name__)


def build_workflow_log(
    state: TicketState,
    start_time: float,
    end_time: float,
    workflow_version: str = "v1.0"
) -> Dict[str, Any]:

    execution_time = end_time - start_time
    executed_at = datetime.fromtimestamp(start_time).isoformat()

    environment = os.getenv("ENVIRONMENT", "production")
    client_id = os.getenv("CLIENT_ID", "unknown_client")

    status = _determine_status(state)
    metrics = _build_metrics(state)
    trace = _build_trace(state)
    metadata = _build_metadata(state)

    log_schema = WorkflowLogSchema(
        # Identification
        client_id=client_id,
        environment=environment,
        workflow_version=workflow_version,

        # Ticket info
        ticket_id=state.get("ticket_id", "unknown"),
        ticket_subject_hash=hash_pii(state.get("ticket_subject", "")),
        executed_at=executed_at,
        execution_time_seconds=round(execution_time, 2),

        # Outcome
        status=status,
        requester_email_hash=hash_pii(state.get("requester_email", "")),
        category=state.get("ticket_category"),
        resolution_status=state.get("resolution_status"),
        customer_type=state.get("customer_type"),

        # Metrics
        metrics=metrics,

        # Error tracking
        workflow_error=state.get("workflow_error"),
        workflow_error_type=state.get("workflow_error_type"),
        workflow_error_node=state.get("workflow_error_node"),
        is_system_error=state.get("is_system_error", False),

        # ✅ PAYLOAD (Swagger-compatible)
        payload={
            "trace": sanitize_trace(trace),
            "final_response": (
                state.get("final_response_public")
                or state.get("draft_response")
            ),
            "private_note": state.get("final_private_note"),
        },

        # Metadata
        metadata=metadata
    )

    log_dict = to_json_safe(log_schema)

    logger.debug(
        f"Built centralized log for ticket {state.get('ticket_id')} "
        f"(status={status}, duration={execution_time:.2f}s)"
    )

    return log_dict


def _determine_status(state: TicketState) -> str:
    """Map workflow outcome → analytics status."""

    if state.get("workflow_error") and state.get("is_system_error"):
        return "ERROR"

    resolution = (state.get("resolution_status") or "").upper()

    if "NEED" in resolution or "INFO" in resolution:
        return "PARTIAL"

    if resolution == "FAILED":
        return "FAILED"

    return "SUCCESS"


def _build_metrics(state: TicketState) -> Dict[str, Any]:
    return {
        "react_iterations": state.get("react_total_iterations", 0),
        "overall_confidence": round(state.get("overall_confidence", 0.0), 3),
        "hallucination_risk": round(state.get("hallucination_risk", 0.0), 3),
        "product_confidence": round(state.get("product_match_confidence", 0.0), 3),
        "customer_type": state.get("customer_type", "END_CUSTOMER"),
        "enough_information": state.get("enough_information", False),
        "needs_more_info": state.get("needs_more_info", False),
        "vision_matches": len(state.get("image_retrieval_results", []) or []),
        "text_matches": len(state.get("text_retrieval_results", []) or []),
        "past_ticket_matches": len(state.get("past_ticket_results", []) or []),
        "planning_confidence": state.get("planning_confidence", 0.0),
        "plan_steps": len(state.get("plan_steps", []) or []),
        "ticket_complexity": state.get("ticket_complexity"),
    }


def _build_trace(state: TicketState) -> Dict[str, Any]:
    return {
        "react_iterations": state.get("react_iterations", []),
        "audit_events": state.get("audit_events", []),
        "retrieval": {
            "vision": state.get("image_retrieval_results", [])[:3],
            "text": state.get("text_retrieval_results", [])[:3],
            "past_ticket_count": len(state.get("past_ticket_results", []) or []),
        },
        "planning": {
            "execution_plan": state.get("execution_plan"),
            "complexity": state.get("ticket_complexity"),
            "confidence": state.get("planning_confidence"),
        },
        "product": {
            "identified": state.get("identified_product"),
            "confidence": state.get("product_match_confidence"),
        },
        "evidence": {
            "decision": state.get("evidence_decision"),
            "needs_more_info": state.get("needs_more_info"),
        },
    }


def _build_metadata(state: TicketState) -> Dict[str, Any]:
    return {
        "attachment_count": len(state.get("ticket_attachments", []) or []),
        "has_images": state.get("has_image", False),
        "ticket_tags": state.get("tags", []),
        "priority": state.get("priority"),
        "ticket_type": state.get("ticket_type"),
        "react_total_iterations": state.get("react_total_iterations", 0),
    }
