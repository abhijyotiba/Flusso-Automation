"""
Routing Agent Node
Classifies ticket using LLM or fallback rules
Enhanced with skip detection for PO/auto-reply tickets
CLEAN + PRODUCTION READY VERSION
"""

import logging
import time
import re
from typing import Dict, Any

from app.graph.state import TicketState
from app.utils.audit import add_audit_event
from app.clients.llm_client import call_llm
from app.config.constants import (
    ROUTING_SYSTEM_PROMPT, 
    SKIP_CATEGORIES,
    PURCHASE_ORDER_NOTE
)

logger = logging.getLogger(__name__)

STEP_NAME = "2️⃣ ROUTING_AGENT"


def _detect_purchase_order(subject: str, text: str, attachments: list) -> bool:
    """
    Fast rule-based detection for Purchase Order tickets.
    These are very common and have clear patterns.
    """
    subject_lower = subject.lower()
    
    # Strong PO indicators in subject
    po_patterns = [
        r'purchase\s*order',
        r'\bpo\s*#?\s*\d+',
        r'\bpo\s*:',
        r'order\s*#?\s*\d+',
        r'\[.*po\]',
        r'p\.o\.',
    ]
    
    for pattern in po_patterns:
        if re.search(pattern, subject_lower):
            # Check if it has PDF attachment (common for POs)
            has_pdf = any(
                att.get('name', '').lower().endswith('.pdf') 
                for att in (attachments or [])
            )
            if has_pdf:
                return True
            # Even without PDF, strong PO subject is enough
            if re.search(r'purchase\s*order', subject_lower):
                return True
    
    return False


def _detect_auto_reply(subject: str, text: str) -> bool:
    """Detect out-of-office and auto-reply messages."""
    combined = (subject + " " + text).lower()
    
    auto_patterns = [
        r'out\s*of\s*(the\s*)?office',
        r'automatic\s*reply',
        r'auto[\s-]?reply',
        r'i\s*am\s*(currently\s*)?(away|out|on\s*vacation)',
        r'will\s*respond\s*when\s*i\s*return',
        r'ooo\s*:',
    ]
    
    for pattern in auto_patterns:
        if re.search(pattern, combined):
            return True
    
    return False


def _check_skip_category(category: str) -> tuple:
    """
    Check if a category should skip the full workflow.
    Returns: (should_skip, skip_reason, skip_private_note)
    """
    if category not in SKIP_CATEGORIES:
        return False, None, None
    
    skip_notes = {
        "purchase_order": PURCHASE_ORDER_NOTE,
        "auto_reply": "🤖 Auto-reply detected - no action needed",
        "spam": "🚫 Spam/promotional email detected - no action needed"
    }
    
    skip_reasons = {
        "purchase_order": "Purchase Order/Invoice received",
        "auto_reply": "Auto-reply/Out of Office message",
        "spam": "Spam or promotional content"
    }
    
    return True, skip_reasons.get(category, f"Skip category: {category}"), skip_notes.get(category, "")


def _determine_rag_requirements(category: str, has_images: bool) -> tuple:
    """
    Determine which RAG pipelines are needed based on category.
    Returns: (requires_vision, requires_text_rag)
    
    Categories:
    - FULL_WORKFLOW (product_issue, replacement_parts, warranty_claim, missing_parts): Both RAGs
    - FLEXIBLE_RAG (product_inquiry, installation_help, finish_color): Text RAG + Vision if images
    - INFORMATION_REQUEST (pricing_request, dealer_inquiry): Text RAG only, no vision
    - SPECIAL (shipping_tracking, return_refund, feedback_suggestion, general): Text RAG only
    """
    from app.config.constants import (
        FULL_WORKFLOW_CATEGORIES, 
        FLEXIBLE_RAG_CATEGORIES,
        INFORMATION_REQUEST_CATEGORIES
    )
    
    if category in FULL_WORKFLOW_CATEGORIES:
        # Always run both for product-related issues
        return True, True
    
    elif category in FLEXIBLE_RAG_CATEGORIES:
        # Text RAG always, vision only if images present
        return has_images, True
    
    elif category in INFORMATION_REQUEST_CATEGORIES:
        # Information lookup only - use Gemini file search, no vision needed
        return False, True
    
    else:
        # Special handling categories - text RAG only
        return False, True


def classify_ticket_category(state: TicketState) -> Dict[str, Any]:
    """
    Classify the ticket category using LLM.
    Falls back to rule-based classification on errors.
    Detects skip categories (PO, auto-reply) for fast processing.
    """
    start_time = time.time()
    
    logger.info(f"{'='*60}")
    logger.info(f"{STEP_NAME} | Starting ticket classification")
    logger.info(f"{'='*60}")

    # -------------------------------------------
    # CHECK IF ALREADY MARKED FOR SKIP (by fetch_ticket)
    # This handles tickets with existing AI tags
    # -------------------------------------------
    if state.get("should_skip") and state.get("ticket_category") == "already_processed":
        logger.info(f"{STEP_NAME} | ⏭️ Ticket already marked for skip (has AI tags)")
        # Return existing state without modification
        return {
            "ticket_category": "already_processed",
            "should_skip": True,
            "skip_reason": state.get("skip_reason", "Already processed by AI"),
            "skip_private_note": "",
            "category_requires_vision": False,
            "category_requires_text_rag": False,
            "audit_events": add_audit_event(
                state,
                event="classify_ticket_category",
                event_type="SKIP",
                details={
                    "category": "already_processed",
                    "reason": "Ticket has existing AI processing tags"
                }
            )["audit_events"]
        }

    subject = state.get("ticket_subject", "") or ""
    text = state.get("ticket_text", "") or ""
    tags = state.get("tags", [])
    ticket_type = state.get("ticket_type")
    attachments = state.get("ticket_attachments", []) or []
    
    logger.info(f"{STEP_NAME} | Input: subject={len(subject)} chars, text={len(text)} chars, tags={tags}")

    # -------------------------------------------
    # FAST PATH: Detect Purchase Orders (rule-based)
    # -------------------------------------------
    if _detect_purchase_order(subject, text, attachments):
        duration = time.time() - start_time
        logger.info(f"{STEP_NAME} | ⚡ Fast detection: Purchase Order ticket")
        logger.info(f"{STEP_NAME} | ✅ Classified as 'purchase_order' (skipping workflow)")
        
        return {
            "ticket_category": "purchase_order",
            "should_skip": True,
            "skip_reason": "Purchase Order - no customer response needed",
            "skip_private_note": PURCHASE_ORDER_NOTE,
            "category_requires_vision": False,
            "category_requires_text_rag": False,
            "audit_events": add_audit_event(
                state,
                event="classify_ticket_category",
                event_type="CLASSIFICATION",
                details={
                    "category": "purchase_order",
                    "reason": "fast_po_detection",
                    "should_skip": True
                }
            )["audit_events"]
        }
    
    # -------------------------------------------
    # FAST PATH: Detect Auto-Reply/OOO
    # -------------------------------------------
    if _detect_auto_reply(subject, text):
        duration = time.time() - start_time
        logger.info(f"{STEP_NAME} | ⚡ Fast detection: Auto-reply/Out of Office")
        logger.info(f"{STEP_NAME} | ✅ Classified as 'auto_reply' (skipping workflow)")
        
        return {
            "ticket_category": "auto_reply",
            "should_skip": True,
            "skip_reason": "Auto-reply/Out of Office message",
            "skip_private_note": "🤖 Auto-reply detected - no action needed",
            "category_requires_vision": False,
            "category_requires_text_rag": False,
            "audit_events": add_audit_event(
                state,
                event="classify_ticket_category",
                event_type="CLASSIFICATION",
                details={
                    "category": "auto_reply",
                    "reason": "fast_auto_reply_detection",
                    "should_skip": True
                }
            )["audit_events"]
        }

    # -------------------------------------------
    # Handle empty tickets
    # -------------------------------------------
    if not subject.strip() and not text.strip():
        logger.warning(f"{STEP_NAME} | ⚠️ Empty ticket content → defaulting to 'general'")
        return {
            "ticket_category": "general",
            "should_skip": False,
            "skip_reason": None,
            "skip_private_note": None,
            "category_requires_vision": False,
            "category_requires_text_rag": True,
            "audit_events": add_audit_event(
                state,
                event="classify_ticket_category",
                event_type="CLASSIFICATION",
                details={"category": "general", "reason": "empty_ticket"}
            )["audit_events"]
        }

    # -------------------------------------------
    # Build prompt content
    # -------------------------------------------
    content = f"Subject: {subject}\n\nDescription:\n{text}"

    if tags:
        content += f"\n\nTags: {', '.join(tags)}"
    if ticket_type:
        content += f"\n\nTicket Type: {ticket_type}"

    try:
        # -------------------------------------------
        # LLM classification
        # -------------------------------------------
        logger.info(f"{STEP_NAME} | Calling LLM for classification...")
        llm_start = time.time()
        
        response = call_llm(
            system_prompt=ROUTING_SYSTEM_PROMPT,
            user_prompt=content,
            response_format="json",
            temperature=0.1
        )
        
        llm_duration = time.time() - llm_start
        logger.info(f"{STEP_NAME} | LLM responded in {llm_duration:.2f}s")

        if not isinstance(response, dict):
            raise ValueError("Invalid LLM response format")

        category = response.get("category", "general")
        confidence = response.get("confidence", 0.0)
        reasoning = response.get("reasoning", "")

        # normalize category
        if not isinstance(category, str) or not category.strip():
            category = "general"

        category = category.lower().strip().replace(" ", "_")

        duration = time.time() - start_time
        logger.info(f"{STEP_NAME} | ✅ Classified as '{category}' (confidence={confidence:.2f})")
        logger.info(f"{STEP_NAME} | Reasoning: {reasoning[:150]}..." if reasoning else f"{STEP_NAME} | No reasoning provided")
        logger.info(f"{STEP_NAME} | Completed in {duration:.2f}s")

        # Check if this category should skip the workflow
        should_skip, skip_reason, skip_note = _check_skip_category(category)
        
        # Determine RAG requirements based on category
        # Check has_image flag (set by fetch_ticket) or ticket_images list
        has_images = state.get("has_image", False) or bool(state.get("ticket_images"))
        requires_vision, requires_text = _determine_rag_requirements(category, has_images)
        
        if should_skip:
            logger.info(f"{STEP_NAME} | 🚀 SKIP WORKFLOW: {skip_reason}")
        else:
            logger.info(f"{STEP_NAME} | 📋 RAG Requirements: vision={requires_vision}, text={requires_text}")

        return {
            "ticket_category": category,
            "should_skip": should_skip,
            "skip_reason": skip_reason,
            "skip_private_note": skip_note,
            "category_requires_vision": requires_vision,
            "category_requires_text_rag": requires_text,
            "audit_events": add_audit_event(
                state,
                event="classify_ticket_category",
                event_type="CLASSIFICATION",
                details={
                    "category": category,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "tags_used": len(tags) > 0,
                    "ticket_type_used": ticket_type is not None,
                    "should_skip": should_skip,
                    "skip_reason": skip_reason,
                    "requires_vision": requires_vision,
                    "requires_text_rag": requires_text
                }
            )["audit_events"]
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"{STEP_NAME} | ❌ LLM classification failed after {duration:.2f}s: {e}", exc_info=True)

        # Fallback
        category = fallback_classification(subject, text, tags)
        logger.warning(f"{STEP_NAME} | Using fallback classification → '{category}'")

        # Check skip and RAG requirements for fallback category too
        should_skip, skip_reason, skip_note = _check_skip_category(category)
        has_images = state.get("has_image", False) or bool(state.get("ticket_images"))
        requires_vision, requires_text = _determine_rag_requirements(category, has_images)

        return {
            "ticket_category": category,
            "should_skip": should_skip,
            "skip_reason": skip_reason,
            "skip_private_note": skip_note,
            "category_requires_vision": requires_vision,
            "category_requires_text_rag": requires_text,
            "audit_events": add_audit_event(
                state,
                event="classify_ticket_category",
                event_type="ERROR",
                details={
                    "category": category,
                    "error": str(e),
                    "fallback_used": True,
                    "should_skip": should_skip,
                    "requires_vision": requires_vision,
                    "requires_text_rag": requires_text
                }
            )["audit_events"]
        }


# -------------------------------------------------------------
# FALLBACK CLASSIFIER (rule-based with scoring)
# -------------------------------------------------------------
def fallback_classification(subject: str, text: str, tags: list) -> str:
    """
    Rule-based fallback when LLM fails. Uses our 16-category system.
    Uses scoring to handle keyword overlap - best match wins.
    """
    content = (subject + " " + text).lower()
    tags_lower = [t.lower() for t in tags]
    
    # Category scores - higher score = better match
    category_scores: Dict[str, float] = {}
    
    # -------------------------------------------
    # SKIP categories (high priority - checked first)
    # -------------------------------------------
    
    # Purchase Order detection (very high weight)
    po_keywords = ["purchase order", "po #", "po#", "p.o.", "invoice", "order confirmation", 
                   "order #", "order number", "payment received", "payment confirmation"]
    po_score = sum(2.0 for kw in po_keywords if kw in content)
    if po_score > 0:
        category_scores["purchase_order"] = po_score + 5  # Boost skip categories
    
    # Auto-reply detection (high priority)
    auto_reply_keywords = ["auto-reply", "auto reply", "automatic reply", "out of office", 
                          "out-of-office", "ooo", "away from office", "on vacation", 
                          "be back", "i am currently out"]
    auto_score = sum(2.0 for kw in auto_reply_keywords if kw in content)
    if auto_score > 0:
        category_scores["auto_reply"] = auto_score + 5  # Boost skip categories
    
    # Spam detection
    spam_keywords = ["unsubscribe", "click here to win", "lottery", "congratulations you won",
                    "act now", "limited time offer", "free money"]
    spam_score = sum(2.0 for kw in spam_keywords if kw in content)
    if spam_score > 0:
        category_scores["spam"] = spam_score + 5  # Boost skip categories
    
    # -------------------------------------------
    # INFORMATION REQUEST categories (high priority - no product ID needed)
    # -------------------------------------------
    
    # Pricing request detection
    pricing_keywords = ["msrp", "price", "pricing", "cost", "how much", "quote", 
                       "price list", "price for", "what does", "wholesale"]
    pricing_score = sum(2.0 for kw in pricing_keywords if kw in content)
    if pricing_score > 0:
        category_scores["pricing_request"] = pricing_score + 3
    
    # Dealer/Partnership inquiry detection
    dealer_keywords = ["dealer", "partnership", "partner", "become a dealer", "distributor",
                      "open account", "credit application", "resale certificate", "wholesale",
                      "dealer application", "flusso partner", "flusso family"]
    dealer_score = sum(2.0 for kw in dealer_keywords if kw in content)
    if dealer_score > 0:
        category_scores["dealer_inquiry"] = dealer_score + 4  # High priority
    
    # -------------------------------------------
    # FULL WORKFLOW categories - tag-based matching
    # -------------------------------------------
    tag_map = {
        "warranty": "warranty_claim",
        "defect": "product_issue",
        "broken": "product_issue",
        "damaged": "product_issue",
        "missing": "missing_parts",
        "replacement": "replacement_parts",
        "return": "return_refund",
        "refund": "return_refund",
        "install": "installation_help",
        "installation": "installation_help",
        "tracking": "shipping_tracking",
        "shipping": "shipping_tracking",
        "feedback": "feedback_suggestion",
        "suggestion": "feedback_suggestion",
        "pricing": "pricing_request",
        "dealer": "dealer_inquiry",
        "partner": "dealer_inquiry"
    }

    # Tags get high weight (they're usually accurate)
    for t in tags_lower:
        for key, cat in tag_map.items():
            if key in t:
                category_scores[cat] = category_scores.get(cat, 0) + 3.0

    # -------------------------------------------
    # Keyword-based matching with weights
    # -------------------------------------------
    keyword_map = {
        # Full workflow (weight 1.0 per match)
        "product_issue": ["broken", "defective", "faulty", "damaged", "not working", 
                         "cracked", "leaking", "leak", "dripping"],
        "replacement_parts": ["replacement part", "spare part", "need part", "order part",
                             "replacement for", "where can i get"],
        "warranty_claim": ["warranty", "guarantee", "covered under", "warranty claim"],
        "missing_parts": ["missing", "not included", "wasn't in the box", "didn't receive",
                         "incomplete", "parts missing"],
        
        # Flexible categories
        "product_inquiry": ["question about", "inquiry", "wondering about", "does this",
                           "what is the", "how does", "is it compatible", "in stock", 
                           "available", "dimensions"],
        "installation_help": ["install", "installation", "setup", "mount", "how to fit",
                             "instructions", "assembly"],
        "finish_color": ["finish", "color", "colour", "chrome", "nickel", "bronze",
                        "brass", "matte", "polished", "brushed"],
        
        # Special handling
        "shipping_tracking": ["tracking", "shipment", "delivery", "where is my order",
                             "when will it arrive", "shipping status"],
        "return_refund": ["return", "refund", "exchange", "money back", "send back"],
        "feedback_suggestion": ["feedback", "suggestion", "recommend", "improve", 
                               "great product", "love it", "disappointed"]
    }

    # Score based on keyword matches
    for category, keywords in keyword_map.items():
        match_count = sum(1.0 for word in keywords if word in content)
        if match_count > 0:
            category_scores[category] = category_scores.get(category, 0) + match_count
    
    # Return the category with highest score, or "general" if no matches
    if not category_scores:
        return "general"
    
    best_category = max(category_scores.items(), key=lambda x: x[1])
    logger.debug(f"Fallback classification scores: {category_scores}, selected: {best_category[0]}")
    return best_category[0]


# -------------------------------------------------------------
# Validator (Optional)
# -------------------------------------------------------------
def validate_routing_result(state: TicketState) -> bool:
    category = state.get("ticket_category")
    return isinstance(category, str) and len(category) > 0
