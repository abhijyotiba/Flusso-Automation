"""
LangGraph Builder with ReACT Agent
Simplified workflow: Fetch → Routing → ReACT Agent → Response → Update
"""

import logging
from typing import Literal
from langgraph.graph import StateGraph, END

from app.graph.state import TicketState

# Import nodes
from app.nodes.fetch_ticket import fetch_ticket_from_freshdesk
from app.nodes.routing_agent import classify_ticket_category
from app.nodes.react_agent import react_agent_loop  # NEW
from app.nodes.customer_lookup import identify_customer_type
from app.nodes.vip_rules import load_vip_rules
from app.nodes.decisions.hallucination_guard import assess_hallucination_risk
from app.nodes.decisions.confidence_check import evaluate_product_confidence
from app.nodes.decisions.vip_compliance import verify_vip_compliance
from app.nodes.response.draft_response import draft_final_response
from app.nodes.response.resolution_logic import decide_tags_and_resolution
from app.nodes.freshdesk_update import update_freshdesk_ticket
from app.nodes.audit_log import write_audit_log
from app.utils.audit import add_audit_event

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
#  SKIP TICKET HANDLER (unchanged)
# ---------------------------------------------------------------------
def skip_ticket_handler(state: TicketState) -> dict:
    """Handle tickets that skip workflow (PO, auto-reply, spam)"""
    category = state.get("ticket_category", "unknown")
    skip_reason = state.get("skip_reason", "Unknown")
    private_note = state.get("private_note", "")
    suggested_tags = state.get("suggested_tags", [])
    
    logger.info(f"[SKIP_HANDLER] Processing skip for category: {category}")
    
    if category == "already_processed":
        return {
            "suggested_tags": [],
            "private_note": "",
            "generated_reply": None,
            "resolution_decision": "already_processed",
            "resolution_reason": skip_reason,
            "skip_workflow_applied": True,
            "audit_events": add_audit_event(
                state,
                event="skip_ticket_handler",
                event_type="SKIP",
                details={"category": category, "reason": "Already processed"}
            )["audit_events"]
        }
    
    tag_map = {
        "purchase_order": ["purchase_order"],
        "auto_reply": ["auto_reply"],
        "spam": ["spam"]
    }
    tags = tag_map.get(category, ["skipped"])
    
    if not private_note:
        note_map = {
            "purchase_order": "📦 Purchase Order received. No response needed.",
            "auto_reply": "🤖 Auto-reply detected. No action needed.",
            "spam": "🚫 Spam detected. No action needed."
        }
        private_note = note_map.get(category, f"Skipped: {skip_reason}")
    
    return {
        "suggested_tags": tags,
        "private_note": private_note,
        "generated_reply": None,
        "resolution_decision": "skip_workflow",
        "resolution_reason": skip_reason,
        "skip_workflow_applied": True,
        "audit_events": add_audit_event(
            state,
            "skip_ticket_handler",
            "SKIP",
            {"category": category, "tags": tags}
        )["audit_events"]
    }


# ---------------------------------------------------------------------
#  ROUTING AFTER CLASSIFICATION
# ---------------------------------------------------------------------
def route_after_routing(state: TicketState) -> Literal["skip_handler", "react_agent"]:
    """
    Route to skip handler or ReACT agent based on classification
    """
    should_skip = state.get("should_skip", False)
    if should_skip:
        category = state.get("ticket_category", "unknown")
        logger.info(f"[ROUTER] Skipping workflow for category: {category}")
        return "skip_handler"
    
    logger.info(f"[ROUTER] Proceeding to ReACT agent")
    return "react_agent"


# ---------------------------------------------------------------------
#  HALLUCINATION GUARD ROUTING (Still validate, but don't block)
# ---------------------------------------------------------------------
def route_after_hallucination_guard(state: TicketState) -> Literal["confidence_check"]:
    """
    Always proceed to confidence check.
    Hallucination risk is considered in resolution_logic.
    """
    from app.config.settings import settings
    
    risk = state.get("hallucination_risk", 0)
    threshold = settings.hallucination_risk_threshold
    
    if risk > threshold:
        logger.warning(f"[ROUTER] High hallucination risk ({risk:.2f}), but continuing to confidence check")
    
    return "confidence_check"


# ---------------------------------------------------------------------
#  BUILD REACT GRAPH
# ---------------------------------------------------------------------
def build_react_graph() -> StateGraph:
    """
    Build LangGraph workflow with ReACT agent.
    
    Simplified flow:
    fetch_ticket → routing → [skip_handler OR react_agent] → 
    customer_lookup → vip_rules → decisions → draft_response → 
    resolution_logic → freshdesk_update → audit_log
    """
    logger.info("[GRAPH_BUILDER] Building ReACT-based workflow...")
    
    graph = StateGraph(TicketState)
    
    # ------------------- ADD NODES -------------------
    graph.add_node("fetch_ticket", fetch_ticket_from_freshdesk)
    graph.add_node("routing", classify_ticket_category)
    graph.add_node("skip_handler", skip_ticket_handler)
    
    # NEW: ReACT Agent (replaces vision/text_rag/past_tickets/orchestration/context_builder)
    graph.add_node("react_agent", react_agent_loop)
    
    graph.add_node("customer_lookup", identify_customer_type)
    graph.add_node("vip_rules", load_vip_rules)
    
    # Keep decision nodes for validation
    graph.add_node("hallucination_guard", assess_hallucination_risk)
    graph.add_node("confidence_check", evaluate_product_confidence)
    graph.add_node("vip_compliance", verify_vip_compliance)
    
    graph.add_node("draft_response", draft_final_response)
    graph.add_node("resolution_logic", decide_tags_and_resolution)
    graph.add_node("freshdesk_update", update_freshdesk_ticket)
    graph.add_node("audit_log", write_audit_log)
    
    # ------------------- ENTRY POINT -------------------
    graph.set_entry_point("fetch_ticket")
    
    # ------------------- BASE FLOW -------------------
    graph.add_edge("fetch_ticket", "routing")
    
    # Route to skip or react agent
    graph.add_conditional_edges(
        "routing",
        route_after_routing,
        {
            "skip_handler": "skip_handler",
            "react_agent": "react_agent"
        }
    )
    
    # Skip handler → directly to freshdesk_update
    graph.add_edge("skip_handler", "freshdesk_update")
    
    # ReACT agent → customer lookup (gather customer context)
    graph.add_edge("react_agent", "customer_lookup")
    graph.add_edge("customer_lookup", "vip_rules")
    
    # After VIP rules → validation gates
    graph.add_edge("vip_rules", "hallucination_guard")
    graph.add_edge("hallucination_guard", "confidence_check")
    graph.add_edge("confidence_check", "vip_compliance")
    
    # After validations → generate response
    graph.add_edge("vip_compliance", "draft_response")
    graph.add_edge("draft_response", "resolution_logic")
    
    # Final chain
    graph.add_edge("resolution_logic", "freshdesk_update")
    graph.add_edge("freshdesk_update", "audit_log")
    graph.add_edge("audit_log", END)
    
    # ------------------- COMPILE -------------------
    compiled_graph = graph.compile()
    logger.info("[GRAPH_BUILDER] ReACT graph compiled successfully ✓")
    
    return compiled_graph
