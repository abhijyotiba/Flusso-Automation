"""
LangGraph Builder with ReACT Agent
Simplified workflow: Fetch → Ticket Extractor → Routing → ReACT Agent → Response → Update
"""

import logging
from typing import Literal
from langgraph.graph import StateGraph, END

from app.graph.state import TicketState

# Import nodes
from app.nodes.fetch_ticket import fetch_ticket_from_freshdesk
from app.nodes.ticket_extractor import extract_ticket_facts  # NEW: Ticket facts extraction
from app.nodes.routing_agent import classify_ticket_category
from app.nodes.react_agent import react_agent_loop  # NEW
from app.nodes.customer_lookup import identify_customer_type
from app.nodes.vip_rules import load_vip_rules
# REMOVED: hallucination_guard and confidence_check (redundant - evidence_resolver handles this)
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
            "purchase_order": "📦 Purchase Order received. No Artificial Intelligence Response Generated",
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
#  HALLUCINATION GUARD ROUTING - REMOVED
#  Evidence resolver now handles confidence assessment
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
#  BUILD REACT GRAPH
# ---------------------------------------------------------------------
def build_react_graph() -> StateGraph:
    """
    Build LangGraph workflow with ReACT agent.
    
    Simplified flow:
    fetch_ticket → ticket_extractor → routing → [skip_handler OR react_agent] → 
    customer_lookup → vip_rules → decisions → draft_response → 
    resolution_logic → freshdesk_update → audit_log
    """
    logger.info("[GRAPH_BUILDER] Building ReACT-based workflow...")
    
    graph = StateGraph(TicketState)
    
    # ------------------- ADD NODES -------------------
    graph.add_node("fetch_ticket", fetch_ticket_from_freshdesk)
    
    # NEW: Ticket Facts Extractor (deterministic extraction before planning)
    graph.add_node("ticket_extractor", extract_ticket_facts)
    
    graph.add_node("routing", classify_ticket_category)
    graph.add_node("skip_handler", skip_ticket_handler)
    
    # NEW: ReACT Agent (replaces vision/text_rag/past_tickets/orchestration/context_builder)
    graph.add_node("react_agent", react_agent_loop)
    
    graph.add_node("customer_lookup", identify_customer_type)
    graph.add_node("vip_rules", load_vip_rules)
    
    # REMOVED: hallucination_guard and confidence_check
    # Evidence resolver (in react_agent) now handles confidence assessment
    graph.add_node("vip_compliance", verify_vip_compliance)
    
    graph.add_node("draft_response", draft_final_response)
    graph.add_node("resolution_logic", decide_tags_and_resolution)
    graph.add_node("freshdesk_update", update_freshdesk_ticket)
    graph.add_node("audit_log", write_audit_log)
    
    # ------------------- ENTRY POINT -------------------
    graph.set_entry_point("fetch_ticket")
    
    # ------------------- BASE FLOW -------------------
    # NEW: fetch_ticket → ticket_extractor → routing
    graph.add_edge("fetch_ticket", "ticket_extractor")
    graph.add_edge("ticket_extractor", "routing")
    
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
    
    # After VIP rules → generate response (simplified flow)
    # REMOVED: hallucination_guard and confidence_check edges
    # Evidence resolver confidence is used directly
    graph.add_edge("vip_rules", "draft_response")
    
    # After response generation → VIP compliance check
    graph.add_edge("draft_response", "vip_compliance")
    
    # VIP compliance → resolution logic
    graph.add_edge("vip_compliance", "resolution_logic")
    # Final chain
    graph.add_edge("resolution_logic", "freshdesk_update")
    graph.add_edge("freshdesk_update", "audit_log")
    graph.add_edge("audit_log", END)
    
    # ------------------- COMPILE -------------------
    compiled_graph = graph.compile()
    logger.info("[GRAPH_BUILDER] ReACT graph compiled successfully ✓")
    
    return compiled_graph
