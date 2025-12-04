"""
LangGraph Builder
Constructs complete workflow with corrected multimodal RAG loop
Includes skip routing for PO/auto-reply/spam tickets
"""

import logging
from typing import Literal
from langgraph.graph import StateGraph, END

from app.graph.state import TicketState

# Import all nodes
from app.nodes.fetch_ticket import fetch_ticket_from_freshdesk
from app.nodes.routing_agent import classify_ticket_category
from app.nodes.vision_pipeline import process_vision_pipeline
from app.nodes.text_rag_pipeline import text_rag_pipeline
from app.nodes.past_tickets import retrieve_past_tickets
from app.nodes.customer_lookup import identify_customer_type
from app.nodes.vip_rules import load_vip_rules
from app.nodes.context_builder import assemble_multimodal_context
from app.nodes.orchestration_agent import orchestration_agent
from app.nodes.decisions.enough_information import check_enough_information
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
#  SKIP TICKET HANDLER NODE
# ---------------------------------------------------------------------
def skip_ticket_handler(state: TicketState) -> dict:
    """
    Handle tickets that should skip the full workflow.
    Categories: purchase_order, auto_reply, spam
    
    Actions:
    - Set appropriate tags
    - Prepare private note
    - Mark as skipped (no public response)
    """
    category = state.get("ticket_category", "unknown")
    skip_reason = state.get("skip_reason", "Category marked for skip")
    skip_note = state.get("skip_private_note", "")
    
    logger.info(f"[SKIP_HANDLER] Processing skip for category: {category}")
    logger.info(f"[SKIP_HANDLER] Reason: {skip_reason}")
    
    # Determine tags based on category - SINGLE TAG only for skip categories
    tag_map = {
        "purchase_order": ["purchase_order"],
        "auto_reply": ["auto_reply"],
        "spam": ["spam"]
    }
    
    tags = tag_map.get(category, ["skipped"])
    
    # Use skip note if provided, otherwise generate one
    if not skip_note:
        note_map = {
            "purchase_order": "📦 Purchase Order/Invoice received. No customer response required.",
            "auto_reply": "🤖 Auto-reply detected. No action needed.",
            "spam": "🚫 Spam/promotional email detected. No action needed."
        }
        skip_note = note_map.get(category, f"Ticket skipped: {skip_reason}")
    
    logger.info(f"[SKIP_HANDLER] Tags: {tags}")
    logger.info(f"[SKIP_HANDLER] Private note: {skip_note}")
    
    return {
        "suggested_tags": tags,
        "private_note": skip_note,
        "generated_reply": None,  # No public response for skipped tickets
        "resolution_decision": "skip_workflow",
        "resolution_reason": skip_reason,
        "skip_workflow_applied": True,
        "audit_events": add_audit_event(
            state,
            event="skip_ticket_handler",
            event_type="SKIP",
            details={
                "category": category,
                "reason": skip_reason,
                "tags": tags,
                "private_note": skip_note
            }
        )["audit_events"]
    }


# ---------------------------------------------------------------------
#  CORRECT MULTIMODAL RAG ROUTER (with skip support)
# ---------------------------------------------------------------------
def route_after_routing(state: TicketState) -> Literal[
    "skip_handler", "vision", "text_rag", "past_tickets", "customer_lookup"
]:
    """
    Multimodal RAG controller with skip support:
        - First check if ticket should skip workflow entirely
        - Then run all relevant pipelines in sequence
        - Only move to customer lookup when ALL applicable RAGs done
    """
    
    # Check for skip first (PO, auto-reply, spam)
    should_skip = state.get("should_skip", False)
    if should_skip:
        category = state.get("ticket_category", "unknown")
        logger.info(f"[ROUTER] 🚀 SKIPPING WORKFLOW for category: {category}")
        return "skip_handler"

    # Check RAG requirements from routing agent
    requires_vision = state.get("category_requires_vision", True)
    requires_text = state.get("category_requires_text_rag", True)
    
    has_image = state.get("has_image", False)
    has_text = state.get("has_text", False)

    ran_vision = state.get("ran_vision", False)
    ran_text = state.get("ran_text_rag", False)
    ran_past = state.get("ran_past_tickets", False)

    # 1. Run vision first if images exist AND category requires it
    if has_image and requires_vision and not ran_vision:
        logger.info("[ROUTER] Running vision pipeline (category requires)")
        return "vision"

    # 2. Then run text RAG if text exists AND category requires it
    if has_text and requires_text and not ran_text:
        logger.info("[ROUTER] Running text RAG pipeline (category requires)")
        return "text_rag"

    # 3. Always check past tickets (once)
    if not ran_past:
        logger.info("[ROUTER] Searching past tickets")
        return "past_tickets"

    # 4. If all RAG done → proceed
    logger.info("[ROUTER] All RAG pipelines completed → customer lookup")
    return "customer_lookup"


# ---------------------------------------------------------------------
#  ORCHESTRATION ROUTING
# ---------------------------------------------------------------------
def route_after_orchestration(
    state: TicketState,
) -> Literal["check_enough_info", "vision", "text_rag", "past_tickets"]:
    """
    If not enough information after orchestration,
    re-run the missing RAG modes before continuing.
    Uses category-based RAG requirements.
    """

    enough = state.get("enough_information", False)
    if enough:
        logger.info("[ROUTER] Orchestration: enough info → proceed to checks")
        return "check_enough_info"

    # Check category-based RAG requirements
    requires_vision = state.get("category_requires_vision", True)
    requires_text = state.get("category_requires_text_rag", True)

    # If not enough info → run missing RAG
    has_image = state.get("has_image", False)
    has_text = state.get("has_text", False)
    ran_vision = state.get("ran_vision", False)
    ran_text = state.get("ran_text_rag", False)
    ran_past = state.get("ran_past_tickets", False)

    if has_image and requires_vision and not ran_vision:
        return "vision"
    if has_text and requires_text and not ran_text:
        return "text_rag"
    if not ran_past:
        return "past_tickets"

    # If all RAG already done
    return "check_enough_info"


# ---------------------------------------------------------------------
#  ENOUGH INFORMATION ROUTING
# ---------------------------------------------------------------------
def route_after_enough_info_check(
    state: TicketState,
) -> Literal["hallucination_guard", "draft_response"]:
    """
    If still not enough info → go straight to draft (ask for more details)
    """
    if state.get("enough_information", False):
        logger.info("[ROUTER] Enough information → hallucination guard")
        return "hallucination_guard"

    logger.info("[ROUTER] Not enough information → draft clarification response")
    return "draft_response"


# ---------------------------------------------------------------------
#  HALLUCINATION GUARD ROUTING
# ---------------------------------------------------------------------
def route_after_hallucination_guard(
    state: TicketState,
) -> Literal["confidence_check", "draft_response"]:
    """
    Always run confidence_check to evaluate product match quality.
    High hallucination risk will still be considered in resolution_logic.
    """
    from app.config.settings import settings
    
    risk = state.get("hallucination_risk", 1.0)
    threshold = settings.hallucination_risk_threshold

    # ALWAYS run confidence check - it's crucial for product matching
    # The resolution_logic node will handle the final decision based on both metrics
    if risk <= threshold:
        logger.info(f"[ROUTER] Low hallucination risk ({risk:.2f} <= {threshold}) → confidence check")
    else:
        logger.info(f"[ROUTER] High hallucination risk ({risk:.2f} > {threshold}) → still running confidence check")
    
    return "confidence_check"


# ---------------------------------------------------------------------
#  BUILD THE FINAL GRAPH
# ---------------------------------------------------------------------
def build_graph() -> StateGraph:
    logger.info("[GRAPH_BUILDER] Building LangGraph workflow...")

    graph = StateGraph(TicketState)

    # ------------------- ADD NODES -------------------
    graph.add_node("fetch_ticket", fetch_ticket_from_freshdesk)
    graph.add_node("routing", classify_ticket_category)
    
    # Skip handler for PO/auto-reply/spam
    graph.add_node("skip_handler", skip_ticket_handler)

    graph.add_node("vision", process_vision_pipeline)
    graph.add_node("text_rag", text_rag_pipeline)
    graph.add_node("past_tickets", retrieve_past_tickets)

    graph.add_node("customer_lookup", identify_customer_type)
    graph.add_node("vip_rules", load_vip_rules)
    graph.add_node("context_builder", assemble_multimodal_context)

    graph.add_node("orchestration", orchestration_agent)
    graph.add_node("check_enough_info", check_enough_information)

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

    # ------------------- MULTIMODAL RAG LOOP (with skip support) -------------------
    graph.add_conditional_edges(
        "routing",
        route_after_routing,
        {
            "skip_handler": "skip_handler",  # NEW: Skip path
            "vision": "vision",
            "text_rag": "text_rag",
            "past_tickets": "past_tickets",
            "customer_lookup": "customer_lookup",
        },
    )
    
    # Skip handler goes directly to freshdesk_update (bypass entire workflow)
    graph.add_edge("skip_handler", "freshdesk_update")

    # After vision/text/past → go back to routing for next step
    graph.add_edge("vision", "routing")
    graph.add_edge("text_rag", "routing")
    graph.add_edge("past_tickets", "routing")

    # ------------------- AFTER ALL RAG -------------------
    graph.add_edge("customer_lookup", "vip_rules")
    graph.add_edge("vip_rules", "context_builder")
    graph.add_edge("context_builder", "orchestration")

    # ------------------- ORCHESTRATION ROUTING -------------------
    graph.add_conditional_edges(
        "orchestration",
        route_after_orchestration,
        {
            "check_enough_info": "check_enough_info",
            "vision": "vision",
            "text_rag": "text_rag",
            "past_tickets": "past_tickets",
        },
    )

    # ------------------- ENOUGH INFO ROUTING -------------------
    graph.add_conditional_edges(
        "check_enough_info",
        route_after_enough_info_check,
        {
            "hallucination_guard": "hallucination_guard",
            "draft_response": "draft_response",
        },
    )

    # ------------------- HALLUCINATION → CONFIDENCE (ALWAYS) -------------------
    # Always run confidence check - it's critical for product matching
    # Resolution logic will consider both hallucination risk AND confidence
    graph.add_edge("hallucination_guard", "confidence_check")

    # ------------------- CONFIDENCE → VIP → DRAFT -------------------
    graph.add_edge("confidence_check", "vip_compliance")
    graph.add_edge("vip_compliance", "draft_response")

    # ------------------- FINAL CHAIN -------------------
    graph.add_edge("draft_response", "resolution_logic")
    graph.add_edge("resolution_logic", "freshdesk_update")
    graph.add_edge("freshdesk_update", "audit_log")
    graph.add_edge("audit_log", END)

    # ------------------- COMPILE -------------------
    compiled_graph = graph.compile()
    logger.info("[GRAPH_BUILDER] Graph compiled successfully")

    return compiled_graph
