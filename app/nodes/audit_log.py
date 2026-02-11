"""
Audit Log Node
Writes complete workflow audit trail as JSON lines.
Also completes the detailed workflow log for examination.

ENHANCED: Now ships logs to centralized collector for analytics.
"""

import logging
import json
import time
from typing import Dict, Any
from pathlib import Path

from app.graph.state import TicketState
from app.utils.detailed_logger import complete_workflow_log, get_current_log
from app.utils.workflow_log_builder import build_workflow_log
from app.utils.log_shipper import  ship_log

logger = logging.getLogger(__name__)
STEP_NAME = "1️⃣7️⃣ AUDIT_LOG"

# Track workflow start time (set by the first node)
_workflow_start_times = {}


def write_audit_log(state: TicketState) -> Dict[str, Any]:
    """
    Final node in the graph.
    Writes the full audit trail + key metrics to a log file.
    
    ENHANCED: Now builds and ships centralized log to remote collector.

    Returns:
        {} (no further state updates)
    """
    node_start = time.time()
    logger.info(f"{STEP_NAME} | ▶ Writing audit trail...")
    
    ticket_id = state.get("ticket_id", "unknown")
    events = state.get("audit_events", []) or []

    logger.info(f"{STEP_NAME} | 📥 Input: ticket_id={ticket_id}, events_count={len(events)}")
    
    # Get workflow start time (should be set by first node)
    workflow_start = _workflow_start_times.get(ticket_id, node_start)
    workflow_end = time.time()

    # Extract vision results for logging
    image_results = state.get("image_retrieval_results", []) or []
    vision_matches = []
    for i, hit in enumerate(image_results[:5]):  # Top 5 matches
        metadata = hit.get("metadata", {})
        vision_matches.append({
            "rank": i + 1,
            "score": round(hit.get("score", 0), 4),
            "product_id": metadata.get("product_id", "N/A"),
            "product_name": metadata.get("product_name", "N/A"),
            "image_name": metadata.get("image_name", "N/A"),
            "category": metadata.get("category", metadata.get("product_category", "N/A"))
        })
    
    # Extract text RAG results
    text_results = state.get("text_retrieval_results", []) or []
    text_matches = []
    for i, hit in enumerate(text_results[:5]):
        text_matches.append({
            "rank": i + 1,
            "score": round(hit.get("score", 0), 4),
            "title": hit.get("metadata", {}).get("title", hit.get("title", "N/A")),
            "content_preview": (hit.get("content", "") or "")[:200]
        })
    
    record = {
        "ticket_id": ticket_id,
        "timestamp": time.time(),
        "resolution_status": state.get("resolution_status"),
        "customer_type": state.get("customer_type"),
        "category": state.get("ticket_category"),
        "overall_confidence": state.get("overall_confidence", 0),
        "metrics": {
            "enough_information": state.get("enough_information"),
            "hallucination_risk": state.get("hallucination_risk"),
            "product_confidence": state.get("product_match_confidence"),
        },
        "retrieval_counts": {
            "text_hits": len(text_results),
            "image_hits": len(image_results),
            "past_ticket_hits": len(state.get("past_ticket_results", []) or []),
        },
        "vision_matches": vision_matches,
        "text_matches": text_matches,
        "events": events,
    }

    try:
        # Write local audit log (existing behavior)
        log_file = Path("audit.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        
        node_duration = time.time() - node_start
        logger.info(f"{STEP_NAME} | ✅ Audit trail written to {log_file} ({len(events)} events) in {node_duration:.2f}s")
        logger.info(f"{STEP_NAME} | 📊 Final summary: status='{record['resolution_status']}', metrics={record['metrics']}")
        
        # Complete and save detailed workflow log
        detailed_log_path = complete_workflow_log(
            resolution_status=record["resolution_status"],
            final_response=state.get("final_response_public") or state.get("draft_response"),
            overall_confidence=state.get("overall_confidence", 0.0),
            metrics=record["metrics"]
        )
        if detailed_log_path:
            logger.info(f"{STEP_NAME} | 📁 Detailed log saved: {detailed_log_path}")
        
        # ==========================================
        # CENTRALIZED LOGGING (NEW)
        # ==========================================
        try:
            logger.info(f"{STEP_NAME} | 📤 Building centralized log...")
            
            # Build the centralized log payload
            log_payload = build_workflow_log(
                state=state,
                start_time=workflow_start,
                end_time=workflow_end,
                workflow_version="v1.0"
            )
            
            # Ship the log (fire-and-forget)
            ship_log(log_payload)
            
            logger.info(f"{STEP_NAME} | ✅ Centralized log sent to collector")
            
        except Exception as ship_error:
            # Logging should NEVER break the workflow
            logger.error(f"{STEP_NAME} | ⚠️ Error preparing centralized log (non-critical): {ship_error}", exc_info=True)
        
        # ==========================================
        
        # Clean up workflow start time tracking
        if ticket_id in _workflow_start_times:
            del _workflow_start_times[ticket_id]
            
    except Exception as e:
        logger.error(f"{STEP_NAME} | ❌ Error writing audit log: {e}", exc_info=True)

    # Final node → no further updates
    return {}


def set_workflow_start_time(ticket_id: str, start_time: float) -> None:
    """
    Set the workflow start time for a ticket.
    Should be called by the first node in the workflow.
    
    Args:
        ticket_id: The ticket ID
        start_time: Unix timestamp when workflow started
    """
    _workflow_start_times[ticket_id] = start_time
