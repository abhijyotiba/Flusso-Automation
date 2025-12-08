"""
Finish Tool - Submit Final Context
Signals the ReACT agent that information gathering is complete
"""

import logging
from typing import Dict, Any, List, Optional, Union
from langchain.tools import tool

logger = logging.getLogger(__name__)


def _normalize_documents(docs: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """Normalize documents list - handles both strings and dicts."""
    if not docs:
        return []
    
    normalized = []
    for doc in docs:
        if isinstance(doc, dict):
            normalized.append(doc)
        elif isinstance(doc, str):
            # Convert string ID to dict format
            normalized.append({"id": doc, "title": doc})
        else:
            normalized.append({"id": str(doc), "title": str(doc)})
    return normalized


def _normalize_images(images: Optional[List[Any]]) -> List[str]:
    """Normalize images list - extracts URLs from dicts or keeps strings."""
    if not images:
        return []
    
    normalized = []
    for img in images:
        if isinstance(img, dict):
            # Extract URL from dict
            url = img.get("url") or img.get("image_url") or img.get("src") or ""
            if url:
                normalized.append(url)
        elif isinstance(img, str):
            normalized.append(img)
    return normalized


def _normalize_tickets(tickets: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """Normalize tickets list - handles both strings and dicts."""
    if not tickets:
        return []
    
    normalized = []
    for ticket in tickets:
        if isinstance(ticket, dict):
            normalized.append(ticket)
        elif isinstance(ticket, str):
            normalized.append({"ticket_id": ticket, "subject": "Unknown"})
        else:
            normalized.append({"ticket_id": str(ticket), "subject": "Unknown"})
    return normalized


@tool
def finish_tool(
    product_identified: bool,
    product_details: Optional[Dict[str, Any]] = None,
    relevant_documents: Optional[List[Any]] = None,
    relevant_images: Optional[List[Any]] = None,
    past_tickets: Optional[List[Any]] = None,
    confidence: float = 0.5,
    reasoning: str = ""
) -> Dict[str, Any]:
    """
    Submit final gathered context and stop the ReACT loop.
    
    Use this tool when:
    - You have enough information to help the customer
    - You've searched all available sources
    - You've exhausted search attempts (no more tools to try)
    - You're confident in the information gathered
    
    Args:
        product_identified: Did you successfully identify the product? (True/False)
        product_details: Product info dict with keys: model, name, category, confidence
        relevant_documents: List of relevant docs (can be dicts or string IDs)
        relevant_images: List of product images (can be URLs or dicts with 'url' key)
        past_tickets: List of similar tickets (can be dicts or ticket IDs)
        confidence: Your confidence (0.0-1.0) in having sufficient information
        reasoning: Explain what you found and any information gaps
    
    Returns:
        {
            "finished": True,
            "summary": str,
            "context_quality": "excellent" | "good" | "fair" | "poor"
        }
    """
    logger.info(f"[FINISH] Product identified: {product_identified}, Confidence: {confidence:.2f}")
    
    # Normalize and validate inputs - handle flexible types
    product_details = product_details or {}
    relevant_documents = _normalize_documents(relevant_documents)
    relevant_images = _normalize_images(relevant_images)
    past_tickets = _normalize_tickets(past_tickets)
    
    # Assess context quality
    context_score = 0
    
    if product_identified:
        context_score += 30
    if len(relevant_documents) >= 3:
        context_score += 25
    elif len(relevant_documents) >= 1:
        context_score += 15
    if len(relevant_images) >= 1:
        context_score += 15
    if len(past_tickets) >= 2:
        context_score += 20
    elif len(past_tickets) >= 1:
        context_score += 10
    if confidence >= 0.8:
        context_score += 10
    
    if context_score >= 80:
        quality = "excellent"
    elif context_score >= 60:
        quality = "good"
    elif context_score >= 40:
        quality = "fair"
    else:
        quality = "poor"
    
    # Build summary
    summary_parts = []
    
    if product_identified:
        model = product_details.get("model", "Unknown")
        name = product_details.get("name", "Product")
        summary_parts.append(f"✓ Product identified: {model} ({name})")
    else:
        summary_parts.append("✗ Product not identified")
    
    summary_parts.append(f"• Found {len(relevant_documents)} relevant document(s)")
    summary_parts.append(f"• Found {len(relevant_images)} product image(s)")
    summary_parts.append(f"• Found {len(past_tickets)} similar ticket(s)")
    summary_parts.append(f"• Confidence: {confidence:.0%}")
    
    if reasoning:
        summary_parts.append(f"• Reasoning: {reasoning}")
    
    summary = "\n".join(summary_parts)
    
    logger.info(f"[FINISH] Context quality: {quality}, Score: {context_score}/100")
    logger.info(f"[FINISH] Summary:\n{summary}")
    
    return {
        "finished": True,
        "product_identified": product_identified,
        "product_details": product_details,
        "relevant_documents": relevant_documents,
        "relevant_images": relevant_images,
        "past_tickets": past_tickets,
        "confidence": confidence,
        "reasoning": reasoning,
        "summary": summary,
        "context_quality": quality,
        "context_score": context_score
    }
