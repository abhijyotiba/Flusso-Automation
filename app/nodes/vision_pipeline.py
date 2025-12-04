"""
Vision Pipeline Node
Processes ticket images and retrieves similar products from Pinecone
With relevance validation to detect category mismatches.
CLEAN + PRODUCTION READY VERSION
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple

from app.graph.state import TicketState, RetrievalHit
from app.utils.audit import add_audit_event
from app.clients.embeddings import embed_image
from app.clients.pinecone_client import get_pinecone_client
from app.clients.llm_client import call_llm
from app.config.settings import settings
from app.utils.detailed_logger import (
    log_node_start, log_node_complete, log_vision_results
)

logger = logging.getLogger(__name__)
STEP_NAME = "3️⃣ VISION_PIPELINE"


# Prompt for validating if vision matches are relevant to the customer's request
VISION_RELEVANCE_PROMPT = """You are a product category relevance validator for a plumbing fixtures support system.

The customer has submitted a support ticket with an image attachment. Our visual search found some product matches.
Your job is to determine if the matched products are RELEVANT to what the customer is asking about.

IMPORTANT: Visual similarity alone is NOT enough. The product CATEGORY must match the customer's request.

Examples of MISMATCHES (category_match = false):
- Customer asks about "shower door hinge" but we found "Sink Faucet" → MISMATCH
- Customer asks about "faucet" but we found "Shower Head" → MISMATCH  
- Customer asks about "toilet" but we found "Bathtub Drain" → MISMATCH

Examples of MATCHES (category_match = true):
- Customer asks about "shower hinge" and we found "Glass Door Hinge" → MATCH
- Customer asks about "faucet handle" and we found "Bathroom Faucet" → MATCH
- Customer asks about "drain" and we found "Shower Drain" → MATCH (same category)

Respond ONLY with valid JSON:
{
    "category_match": true/false,
    "customer_needs": "<what product category the customer is asking about>",
    "matched_category": "<what product category was found by visual search>",
    "reasoning": "<brief explanation of match/mismatch>",
    "quality": "HIGH" | "LOW" | "CATEGORY_MISMATCH"
}

Where quality is:
- "HIGH": Product category matches customer's request perfectly
- "LOW": Related category but uncertain if it's what customer needs
- "CATEGORY_MISMATCH": Completely different product category than what customer asked about"""


def _validate_vision_relevance(
    state: TicketState, 
    hits: List[RetrievalHit]
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Use LLM to validate if vision matches are relevant to the customer's request.
    
    Returns:
        Tuple of (quality, reason, matched_category, expected_category)
    """
    if not hits:
        return ("NO_MATCH", "No visual matches found in product catalog", None, None)
    
    # Check minimum similarity threshold
    top_score = hits[0].get("score", 0) if hits else 0
    min_threshold = settings.vision_min_similarity_threshold
    
    if top_score < min_threshold:
        logger.info(f"{STEP_NAME} | ⚠ Top score {top_score:.3f} below threshold {min_threshold}")
        return (
            "LOW", 
            f"Visual similarity score ({top_score:.2f}) is below confidence threshold ({min_threshold})",
            None, None
        )
    
    # Skip LLM validation if disabled
    if not settings.vision_category_validation:
        return ("HIGH", "Category validation disabled - accepting visual matches", None, None)
    
    # Build context for LLM validation
    ticket_subject = state.get("ticket_subject", "") or ""
    ticket_text = state.get("ticket_text", "") or ""
    
    # Extract top matches info for LLM
    top_matches = []
    for i, hit in enumerate(hits[:3], 1):
        meta = hit.get("metadata", {}) or {}
        top_matches.append({
            "rank": i,
            "score": hit.get("score", 0),
            "product_name": meta.get("product_title", meta.get("product_name", "Unknown")),
            "category": meta.get("product_category", meta.get("category", "Unknown")),
            "sub_category": meta.get("sub_category", ""),
            "model": meta.get("model_no", "N/A")
        })
    
    user_prompt = f"""CUSTOMER TICKET:
Subject: {ticket_subject}
Description: {ticket_text[:500]}

TOP VISUAL MATCHES FROM OUR CATALOG:
{json.dumps(top_matches, indent=2)}

Is the product category of these matches relevant to what the customer is asking about?"""

    try:
        logger.info(f"{STEP_NAME} | 🔄 Validating category relevance with LLM...")
        validation_start = time.time()
        
        response = call_llm(
            system_prompt=VISION_RELEVANCE_PROMPT,
            user_prompt=user_prompt,
            response_format="json"
        )
        
        validation_duration = time.time() - validation_start
        logger.info(f"{STEP_NAME} | ✓ Validation complete in {validation_duration:.2f}s")
        
        # Parse response
        if isinstance(response, str):
            result = json.loads(response)
        else:
            result = response
        
        quality = result.get("quality", "LOW")
        reasoning = result.get("reasoning", "No reasoning provided")
        customer_needs = result.get("customer_needs", "")
        matched_category = result.get("matched_category", "")
        
        logger.info(f"{STEP_NAME} | 🎯 Relevance: quality={quality}")
        logger.info(f"{STEP_NAME} | 📝 Reason: {reasoning}")
        
        if quality == "CATEGORY_MISMATCH":
            logger.warning(f"{STEP_NAME} | ⚠ Category mismatch detected!")
            logger.warning(f"{STEP_NAME} |   Customer needs: {customer_needs}")
            logger.warning(f"{STEP_NAME} |   But matched: {matched_category}")
        
        return (quality, reasoning, matched_category, customer_needs)
        
    except Exception as e:
        logger.error(f"{STEP_NAME} | ❌ Relevance validation failed: {e}")
        # Default to accepting matches if validation fails
        return ("LOW", f"Validation error: {str(e)}", None, None)


def process_vision_pipeline(state: TicketState) -> Dict[str, Any]:
    """
    Step:
        1. Embed ticket images using CLIP (or Vertex AI embeddings)
        2. Query Pinecone image index
        3. Validate if matches are relevant to customer's request
        4. Return ranked RetrievalHit list with quality assessment

    Returns:
        Partial state update:
            - image_retrieval_results
            - vision_match_quality
            - vision_relevance_reason
            - vision_matched_category
            - vision_expected_category
            - ran_vision = True
            - audit_events
    """

    start_time = time.time()
    logger.info(f"{STEP_NAME} | ▶ Starting vision pipeline")
    
    # Start node log
    node_log = log_node_start("vision_pipeline", {"image_count": len(state.get("ticket_images", []))})
    
    images = state.get("ticket_images", [])
    logger.info(f"{STEP_NAME} | 📥 Input: {len(images)} image(s) to process")

    if not images:
        duration = time.time() - start_time
        logger.info(f"{STEP_NAME} | ⏭ No images found - skipping ({duration:.2f}s)")
        return {
            "image_retrieval_results": [],
            "vision_match_quality": "NO_MATCH",
            "vision_relevance_reason": "No images attached to ticket",
            "ran_vision": True,
            "audit_events": add_audit_event(
                state,
                "vision_pipeline",
                "INFO",
                {"image_count": 0, "results_count": 0, "quality": "NO_MATCH"}
            )["audit_events"]
        }

    logger.info(f"{STEP_NAME} | 🔄 Processing {len(images)} image(s)...")

    client = get_pinecone_client()
    top_k = settings.image_retrieval_top_k

    all_hits: List[RetrievalHit] = []

    for idx, img_url in enumerate(images, 1):
        try:
            logger.info(f"{STEP_NAME} | 🖼 [{idx}/{len(images)}] Embedding image: {img_url}")
            embed_start = time.time()

            vector = embed_image(img_url)
            embed_duration = time.time() - embed_start

            if not vector:
                logger.warning(f"{STEP_NAME} | ⚠ [{idx}] Failed embedding for: {img_url}")
                continue
            
            logger.info(f"{STEP_NAME} | ✓ [{idx}] Embedded in {embed_duration:.2f}s, vector dim={len(vector)}")

            query_start = time.time()
            hits = client.query_images(vector=vector, top_k=top_k)
            query_duration = time.time() - query_start
            
            logger.info(f"{STEP_NAME} | 🔍 [{idx}] Pinecone query returned {len(hits)} hits in {query_duration:.2f}s")
            all_hits.extend(hits)

        except Exception as e:
            logger.error(f"{STEP_NAME} | ❌ [{idx}] Error processing {img_url}: {e}", exc_info=True)

    # Deduplicate + sort
    all_hits = sorted(all_hits, key=lambda h: h["score"], reverse=True)

    # If multiple images → keep a reasonable number
    limit = max(top_k, len(images) * top_k)
    all_hits = all_hits[:limit]

    # === NEW: Validate relevance of vision matches ===
    vision_quality, vision_reason, matched_cat, expected_cat = _validate_vision_relevance(state, all_hits)
    
    duration = time.time() - start_time
    top_scores = [f"{h.get('score', 0):.3f}" for h in all_hits[:3]]
    logger.info(f"{STEP_NAME} | ✅ Complete: {len(all_hits)} matches in {duration:.2f}s")
    logger.info(f"{STEP_NAME} | 📤 Top scores: {top_scores}")
    logger.info(f"{STEP_NAME} | 🎯 Match quality: {vision_quality}")
    
    # Log detailed vision results for examination
    log_vision_results(node_log, all_hits)
    log_node_complete(
        node_log,
        output_summary={
            "total_matches": len(all_hits),
            "top_scores": top_scores,
            "duration_seconds": duration,
            "vision_quality": vision_quality,
            "vision_reason": vision_reason
        },
        retrieval_results=[{
            "rank": i+1,
            "score": h.get("score", 0),
            "product_id": h.get("metadata", {}).get("product_id", "N/A"),
            "product_name": h.get("metadata", {}).get("product_name", "N/A"),
            "image_name": h.get("metadata", {}).get("image_name", "N/A"),
            "category": h.get("metadata", {}).get("category", "N/A"),
            "full_metadata": h.get("metadata", {})
        } for i, h in enumerate(all_hits)]
    )

    return {
        "image_retrieval_results": all_hits,
        "vision_match_quality": vision_quality,
        "vision_relevance_reason": vision_reason,
        "vision_matched_category": matched_cat,
        "vision_expected_category": expected_cat,
        "ran_vision": True,
        "audit_events": add_audit_event(
            state,
            "vision_pipeline",
            "SUCCESS",
            {
                "image_count": len(images),
                "results_count": len(all_hits),
                "duration_seconds": duration,
                "vision_quality": vision_quality,
                "vision_reason": vision_reason
            }
        )["audit_events"]
    }