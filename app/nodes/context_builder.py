"""
Context Builder Node
Combines all retrieval results into unified context for LLM.
With smart filtering for vision match quality.
"""

import logging
import time
from typing import Dict, Any, List

from app.graph.state import TicketState, RetrievalHit

logger = logging.getLogger(__name__)
STEP_NAME = "8️⃣ CONTEXT_BUILDER"


def assemble_multimodal_context(state: TicketState) -> Dict[str, Any]:
    """
    Combine all retrieval results (text, image, past tickets, VIP rules)
    into a single formatted context string for the orchestration agent.
    
    Smart handling of vision results:
    - HIGH quality: Include as confident product matches
    - LOW quality: Include with uncertainty warning
    - CATEGORY_MISMATCH: Exclude or mark as irrelevant
    - NO_MATCH: Skip vision section
    """
    start_time = time.time()
    logger.info(f"{STEP_NAME} | ▶ Building unified context...")

    sections: List[str] = []

    # ---------------- TEXT RAG ----------------
    text_hits: List[RetrievalHit] = state.get("text_retrieval_results", []) or []
    if text_hits:
        sections.append("### PRODUCT DOCUMENTATION\n")
        for i, hit in enumerate(text_hits[:5], 1):
            title = hit.get("metadata", {}).get("title", f"Document {i}")
            content = (hit.get("content") or "")[:500]
            score = hit.get("score", 0.0)
            sections.append(
                f"{i}. **{title}** (score: {score:.2f})\n{content}\n"
            )

    # ---------------- IMAGE RAG (with quality filtering) ----------------
    image_hits: List[RetrievalHit] = state.get("image_retrieval_results", []) or []
    vision_quality = state.get("vision_match_quality", "LOW")
    vision_reason = state.get("vision_relevance_reason", "")
    vision_matched_cat = state.get("vision_matched_category", "")
    vision_expected_cat = state.get("vision_expected_category", "")
    
    if image_hits:
        if vision_quality == "HIGH":
            # Confident match - include as reliable
            sections.append("\n### PRODUCT MATCHES (VISUAL) ✅ HIGH CONFIDENCE\n")
            sections.append("*These products match the customer's attached image and request.*\n")
            _add_image_hits_to_context(sections, image_hits)
            
        elif vision_quality == "LOW":
            # Uncertain - include with warning
            sections.append("\n### PRODUCT MATCHES (VISUAL) ⚠️ LOW CONFIDENCE\n")
            sections.append(f"*Note: {vision_reason}*\n")
            sections.append("*These are visually similar products but relevance is uncertain. Verify before using.*\n")
            _add_image_hits_to_context(sections, image_hits)
            
        elif vision_quality == "CATEGORY_MISMATCH":
            # Wrong category - warn the LLM not to use these
            sections.append("\n### VISUAL SEARCH RESULTS ❌ CATEGORY MISMATCH\n")
            sections.append(f"**WARNING: The customer is asking about '{vision_expected_cat}' but visual search found '{vision_matched_cat}'.**\n")
            sections.append("**DO NOT use these product matches in your response - they are the wrong product category.**\n")
            sections.append(f"*Reason: {vision_reason}*\n")
            # Still list them so agent can see, but clearly marked as wrong
            sections.append("\n*Irrelevant matches (for reference only):*\n")
            for i, hit in enumerate(image_hits[:2], 1):
                meta = hit.get("metadata", {}) or {}
                product_title = meta.get("product_title", "Unknown Product")
                category = meta.get("product_category", "Unknown Category")
                sections.append(f"  {i}. ~~{product_title}~~ (Category: {category}) - NOT RELEVANT\n")
                
        elif vision_quality == "NO_MATCH":
            # No useful matches
            sections.append("\n### VISUAL SEARCH RESULTS ❌ NO MATCH\n")
            sections.append("*Could not identify the product from the attached image. Ask customer for more details.*\n")
            if vision_reason:
                sections.append(f"*Reason: {vision_reason}*\n")
    
    # Log vision handling
    logger.info(f"{STEP_NAME} | 🖼 Vision quality: {vision_quality}")
    if vision_quality == "CATEGORY_MISMATCH":
        logger.warning(f"{STEP_NAME} | ⚠ Vision mismatch: expected '{vision_expected_cat}', got '{vision_matched_cat}'")

    # ---------------- PAST TICKETS ----------------
    past_hits: List[RetrievalHit] = state.get("past_ticket_results", []) or []
    if past_hits:
        sections.append("\n### SIMILAR PAST TICKETS\n")
        for i, hit in enumerate(past_hits[:3], 1):
            meta = hit.get("metadata", {}) or {}
            ticket_id = meta.get("ticket_id", "Unknown")
            resolution_type = meta.get("resolution_type", "N/A")
            content = (hit.get("content") or "")[:300]
            score = hit.get("score", 0.0)

            sections.append(
                f"{i}. Ticket #{ticket_id} ({resolution_type}) "
                f"- Similarity: {score:.2f}\n{content}\n"
            )

    # ---------------- VIP RULES ----------------
    vip_rules = state.get("vip_rules", {}) or {}
    if vip_rules:
        sections.append("\n### VIP CUSTOMER RULES\n")
        for key, value in vip_rules.items():
            label = key.replace("_", " ").title()
            sections.append(f"- {label}: {value}\n")

    multimodal_context = (
        "\n".join(sections) if sections else "No relevant context found."
    )

    duration = time.time() - start_time
    logger.info(f"{STEP_NAME} | 📊 Sources: text={len(text_hits)}, image={len(image_hits)}, past={len(past_hits)}, vip_rules={bool(vip_rules)}")
    logger.info(f"{STEP_NAME} | ✅ Complete: {len(multimodal_context)} chars context in {duration:.2f}s")

    # Append audit info (manual, consistent with your pattern)
    audit_events = state.get("audit_events", []) or []
    audit_events.append(
        {
            "event": "assemble_multimodal_context",
            "text_hits": len(text_hits),
            "image_hits": len(image_hits),
            "past_hits": len(past_hits),
            "has_vip_rules": bool(vip_rules),
            "vision_quality": vision_quality,
            "context_length": len(multimodal_context),
            "duration_seconds": duration,
        }
    )

    return {
        "multimodal_context": multimodal_context,
        "audit_events": audit_events,
    }


def _add_image_hits_to_context(sections: List[str], image_hits: List[RetrievalHit]) -> None:
    """Helper to add image hits to context sections."""
    for i, hit in enumerate(image_hits[:5], 1):
        meta = hit.get("metadata", {}) or {}
        product_title = meta.get("product_title", "Unknown Product")
        model_no = meta.get("model_no", "N/A")
        finish = meta.get("finish", "N/A")
        category = meta.get("product_category", "Unknown Category")
        sub_category = meta.get("sub_category", "")
        collection = meta.get("collection", "")
        score = hit.get("score", 0.0)
        
        category_info = f"{category}"
        if sub_category:
            category_info += f" > {sub_category}"
        
        sections.append(
            f"{i}. **{product_title}** (Model: {model_no}, Finish: {finish})\n"
            f"   Category: {category_info}\n"
            f"   Collection: {collection}\n"
            f"   Visual Similarity: {score:.2f}\n"
        )
