"""
Product Match Confidence Check Node
Evaluates confidence in product identification.
Factors in vision match quality for better accuracy.
"""

import logging
import time
from typing import Dict, Any

from app.graph.state import TicketState
from app.utils.audit import add_audit_event
from app.clients.llm_client import call_llm
from app.config.constants import PRODUCT_CONFIDENCE_PROMPT
from app.utils.detailed_logger import (
    log_node_start, log_node_complete, log_llm_interaction
)

logger = logging.getLogger(__name__)
STEP_NAME = "1️⃣2️⃣ CONFIDENCE_CHECK"


def evaluate_product_confidence(state: TicketState) -> Dict[str, Any]:
    """
    Evaluate product match confidence using LLM output.
    
    Also factors in vision_match_quality:
    - CATEGORY_MISMATCH → force low confidence for image-based tickets
    - NO_MATCH → reduce confidence if ticket has images
    
    SMART OVERRIDE: 
    - If text-based confidence is high (>0.8), vision failures are ignored.

    Writes:
      - product_match_confidence: float in [0, 1]
      - product_match_reasoning: str
    """
    start_time = time.time()
    logger.info(f"{STEP_NAME} | ▶ Evaluating product match confidence...")
    
    # Initialize defaults
    confidence = 0.5
    original_confidence = 0.5
    reasoning = "Processing..."
    
    # Start node log
    node_log = log_node_start("confidence_check", {})

    ticket_text = state.get("ticket_text", "") or ""
    # Ensure context is string to prevent type errors
    raw_context = state.get("multimodal_context", "")
    context = str(raw_context) if raw_context else ""
    
    # Get vision quality info
    vision_quality = state.get("vision_match_quality", "NO_MATCH")
    vision_reason = state.get("vision_relevance_reason", "")
    has_images = bool(state.get("ticket_images", []))
    
    node_log.input_summary = {
        "ticket_length": len(ticket_text),
        "context_length": len(context),
        "vision_quality": vision_quality,
        "has_images": has_images
    }
    
    logger.info(f"{STEP_NAME} | 📥 Input: ticket_len={len(ticket_text)}, context_len={len(context)}")
    logger.info(f"{STEP_NAME} | 🖼 Vision quality: {vision_quality}")

    # Add vision quality context to the prompt
    vision_context = ""
    if has_images:
        vision_context = f"""

VISION ANALYSIS RESULT:
- Match Quality: {vision_quality}
- Reason: {vision_reason}

IMPORTANT: If vision quality is 'CATEGORY_MISMATCH' or 'NO_MATCH', the visual product identification 
failed. However, if the text/document evidence is STRONG, you may still assign high confidence.
"""

    user_prompt = f"""Ticket:
{ticket_text}

Retrieved Product Information:
{context}
{vision_context}
"""

    try:
        logger.info(f"{STEP_NAME} | 🔄 Calling LLM for confidence assessment...")
        llm_start = time.time()
        
        response = call_llm(
            system_prompt=PRODUCT_CONFIDENCE_PROMPT,
            user_prompt=user_prompt,
            response_format="json",
        )
        
        llm_duration = time.time() - llm_start
        logger.info(f"{STEP_NAME} | ✓ LLM response in {llm_duration:.2f}s")

        if not isinstance(response, dict):
            logger.warning(f"{STEP_NAME} | ⚠ Non-dict response from LLM, using default 0.5")
            confidence = 0.5
            reasoning = "LLM returned invalid format. Defaulting to 0.5."
        else:
            confidence = float(response.get("confidence", 0.5))
            reasoning = response.get("reasoning", "No reasoning provided.")

        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        original_confidence = confidence
        
        # === VISION QUALITY OVERRIDE (IMPROVED) ===
        # Logic: If we found the exact manual (High Text Confidence), 
        # we shouldn't care if the Vision Search failed.
        
        vision_cap_applied = False
        max_confidence = 1.0
        
        # Threshold for "Strong Text Evidence"
        has_strong_text_evidence = confidence > 0.8

        if has_images:
            if not has_strong_text_evidence:
                # Only apply penalties if we are NOT sure based on text
                if vision_quality == "CATEGORY_MISMATCH":
                    max_confidence = 0.3
                elif vision_quality == "NO_MATCH":
                    max_confidence = 0.5  # Bumped up from 0.4 to be less aggressive
                elif vision_quality == "LOW":
                    max_confidence = 0.7
                
                if confidence > max_confidence:
                    confidence = max_confidence
                    vision_cap_applied = True
                    logger.info(f"{STEP_NAME} | ⚠ Vision {vision_quality} & Weak Text - capping confidence from {original_confidence:.2f} to {confidence:.2f}")
                    reasoning += f" [System Note: Confidence capped at {max_confidence} due to Vision Quality: {vision_quality}]"
            else:
                logger.info(f"{STEP_NAME} | 🛡️ Vision {vision_quality} ignored due to strong text evidence (Conf: {confidence:.2f})")

        duration = time.time() - start_time
        logger.info(f"{STEP_NAME} | 🎯 Decision: product_match_confidence={confidence:.2f}")
        if reasoning:
            logger.info(f"{STEP_NAME} | 📝 Reasoning: {reasoning[:150]}..." if len(str(reasoning)) > 150 else f"{STEP_NAME} | 📝 Reasoning: {reasoning}")
        logger.info(f"{STEP_NAME} | ✅ Complete in {duration:.2f}s")
        
        # Log LLM interaction
        log_llm_interaction(
            node_log,
            system_prompt=PRODUCT_CONFIDENCE_PROMPT,
            user_prompt=user_prompt,
            response=str(response),
            parsed_response=response if isinstance(response, dict) else {"confidence": confidence}
        )
        log_node_complete(
            node_log,
            output_summary={
                "product_match_confidence": confidence,
                "original_confidence": original_confidence,
                "vision_quality": vision_quality
            },
            decision={"confidence": confidence, "vision_adjusted": vision_cap_applied},
            reasoning=str(reasoning) if reasoning else None
        )

        audit_events = add_audit_event(
            state,
            event="evaluate_product_confidence",
            event_type="DECISION",
            details={
                "confidence": confidence, 
                "original_confidence": original_confidence,
                "vision_quality": vision_quality,
                "reasoning": reasoning, 
                "llm_duration_seconds": llm_duration,
                "vision_cap_applied": vision_cap_applied
            },
        )["audit_events"]

        return {
            "product_match_confidence": confidence,
            "product_match_reasoning": reasoning, # Important for downstream context
            "audit_events": audit_events,
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"{STEP_NAME} | ❌ Error after {duration:.2f}s: {e}", exc_info=True)

        # For fallback, consider vision quality
        if has_images and vision_quality in ["CATEGORY_MISMATCH", "NO_MATCH"]:
            fallback_conf = 0.15  # Very low if vision failed
        else:
            fallback_conf = 0.3  # safe, low-confidence default
            
        fallback_reasoning = f"Error in confidence check: {str(e)}. Using fallback confidence."
        logger.warning(f"{STEP_NAME} | Using fallback confidence={fallback_conf}")

        audit_events = add_audit_event(
            state,
            event="evaluate_product_confidence",
            event_type="ERROR",
            details={"error": str(e), "confidence": fallback_conf, "vision_quality": vision_quality},
        )["audit_events"]

        return {
            "product_match_confidence": fallback_conf,
            "product_match_reasoning": fallback_reasoning,
            "audit_events": audit_events,
        }