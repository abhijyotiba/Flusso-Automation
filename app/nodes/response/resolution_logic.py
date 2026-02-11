"""
Resolution Logic Node
Determines final status and tags based on evidence resolver analysis.
Simplified: Uses evidence_resolver's confidence instead of separate hallucination/confidence nodes.
"""

import logging
import time
from typing import Dict, Any, List

from app.graph.state import TicketState
from app.utils.audit import add_audit_event
from app.config.constants import ResolutionStatus
from app.config.settings import settings

logger = logging.getLogger(__name__)
STEP_NAME = "1️⃣5️⃣ RESOLUTION_LOGIC"


def decide_tags_and_resolution(state: TicketState) -> Dict[str, Any]:
    """
    Determine resolution status and tags.
    
    Uses evidence_resolver's confidence (from react_agent) instead of
    separate hallucination_guard and confidence_check nodes.

    Returns:
        Partial update with:
            - resolution_status
            - extra_tags
            - final_response_public
            - audit_events
    """
    start_time = time.time()
    logger.info(f"{STEP_NAME} | ▶ Determining final resolution status...")

    # === Check for system errors first ===
    is_system_error = state.get("is_system_error", False)
    workflow_error = state.get("workflow_error")
    
    if is_system_error or workflow_error:
        logger.error(f"{STEP_NAME} | 🚨 SYSTEM ERROR detected - marking for manual review")
        tags: List[str] = ["SYSTEM_ERROR", "NEEDS_HUMAN_REVIEW", "MANUAL_REQUIRED"]
        
        duration = time.time() - start_time
        return {
            "resolution_status": "SYSTEM_ERROR",
            "extra_tags": tags,
            "final_response_public": state.get("draft_response", ""),  # Will contain error note
            "audit_events": add_audit_event(
                state,
                event="resolution_logic",
                event_type="SYSTEM_ERROR",
                details={
                    "decision_reason": f"System error: {workflow_error}",
                    "error_type": state.get("workflow_error_type"),
                    "duration_seconds": duration,
                }
            )["audit_events"],
        }

    enough_info = state.get("enough_information", False)
    customer_type = state.get("customer_type", "END_CUSTOMER")
    
    # Get evidence resolver decision (primary source of truth)
    needs_more_info = state.get("needs_more_info", False)
    evidence_analysis = state.get("evidence_analysis", {})
    evidence_action = evidence_analysis.get("resolution_action", "proceed") if evidence_analysis else "proceed"
    evidence_confidence = evidence_analysis.get("final_confidence", 0.0) if evidence_analysis else 0.0
    
    # Use evidence resolver's confidence, fallback to product_match_confidence from react_agent
    confidence = state.get("product_match_confidence", evidence_confidence)
    # Use evidence resolver's assessment for risk (inverted confidence = risk)
    # If evidence is low confidence, that implies higher risk
    risk = max(0.0, 1.0 - evidence_confidence) if evidence_confidence > 0 else state.get("hallucination_risk", 0.3)
    
    logger.info(f"{STEP_NAME} | 📥 Input metrics:")
    logger.info(f"{STEP_NAME} |   - needs_more_info: {needs_more_info}")
    logger.info(f"{STEP_NAME} |   - evidence_action: {evidence_action}")
    logger.info(f"{STEP_NAME} |   - evidence_confidence: {evidence_confidence:.2f}")
    logger.info(f"{STEP_NAME} |   - enough_information: {enough_info}")
    logger.info(f"{STEP_NAME} |   - product_confidence: {confidence:.2f} (threshold: {settings.product_confidence_threshold})")
    logger.info(f"{STEP_NAME} |   - derived_risk: {risk:.2f}")
    logger.info(f"{STEP_NAME} |   - customer_type: {customer_type}")

    tags: List[str] = list(state.get("extra_tags", []) or [])
    draft = state.get("draft_response", "") or ""

    status = ResolutionStatus.RESOLVED.value
    decision_reason = ""

    # ⚠️ PRIORITY 0: Evidence resolver says we need more info from customer
    # This MUST be checked first - if evidence_resolver detected conflicting 
    # evidence or low confidence, we should NOT mark as RESOLVED.
    if needs_more_info or evidence_action == "request_info":
        status = ResolutionStatus.NEEDS_MORE_INFO.value
        tags.extend(["NEEDS_MORE_INFO", "AWAITING_CUSTOMER_REPLY"])
        decision_reason = f"evidence resolver requested more info (action={evidence_action}, confidence={evidence_confidence:.2f})"
        logger.info(f"{STEP_NAME} | 📨 Status: NEEDS_MORE_INFO - {decision_reason}")

    # Priority 1: Not enough information gathered
    elif not enough_info:
        status = ResolutionStatus.AI_UNRESOLVED.value
        tags.extend(["AI_UNRESOLVED", "NEEDS_HUMAN_REVIEW"])
        decision_reason = f"insufficient information gathered (enough_info={enough_info})"
        logger.info(f"{STEP_NAME} | 🚨 Status: AI_UNRESOLVED - {decision_reason}")

    # Priority 2: Low product confidence
    elif confidence < settings.product_confidence_threshold:
        status = ResolutionStatus.LOW_CONFIDENCE_MATCH.value
        tags.extend(["LOW_CONFIDENCE_MATCH", "NEEDS_HUMAN_REVIEW"])
        decision_reason = f"low product confidence ({confidence:.2f} < {settings.product_confidence_threshold})"
        logger.info(f"{STEP_NAME} | ⚠️ Status: LOW_CONFIDENCE_MATCH - {decision_reason}")

    # All checks passed - can mark as resolved
    else:
        status = ResolutionStatus.RESOLVED.value
        tags.append("AI_PROCESSED")
        decision_reason = "all checks passed (evidence confirmed, low risk, high confidence)"
        logger.info(f"{STEP_NAME} | ✅ Status: RESOLVED - {decision_reason}")

    # Remove duplicates
    tags = list(set(tags))

    duration = time.time() - start_time
    logger.info(f"{STEP_NAME} | 🎯 Final decision: status='{status}', tags={tags}")
    logger.info(f"{STEP_NAME} | ✅ Complete in {duration:.2f}s")

    audit_events = add_audit_event(
        state,
        event="decide_tags_and_resolution",
        event_type="DECISION",
        details={
            "resolution_status": status,
            "decision_reason": decision_reason,
            "tags": tags,
            "needs_more_info": needs_more_info,
            "evidence_action": evidence_action,
            "evidence_confidence": evidence_confidence,
            "enough_info": enough_info,
            "product_confidence": confidence,
            "customer_type": customer_type,
        },
    )["audit_events"]

    return {
        "resolution_status": status,
        "extra_tags": tags,
        "final_response_public": draft,
        "audit_events": audit_events,
    }
