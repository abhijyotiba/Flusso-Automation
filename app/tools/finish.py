"""
Finish Tool - FIXED VERSION
More lenient with input types and better validation
"""

import logging
from typing import Dict, Any, List, Optional, Union
from langchain.tools import tool

logger = logging.getLogger(__name__)


def _safe_extract_list(value: Any, default_type: str = "str") -> List[Any]:
    """Safely extract a list from various input types"""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, dict):
        return [value]
    return []


def _normalize_product_details(value: Any) -> Dict[str, Any]:
    """
    Normalize product_details to always be a dict.
    Handles: None, dict, list of dicts
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        # If it's a list, take the first item (primary product)
        # or merge multiple products into one response
        if len(value) == 0:
            return {}
        elif len(value) == 1:
            return value[0] if isinstance(value[0], dict) else {}
        else:
            # Multiple products - take first as primary, note others
            primary = value[0] if isinstance(value[0], dict) else {}
            # Store additional products in the dict
            additional = [p for p in value[1:] if isinstance(p, dict)]
            if additional:
                primary["additional_products"] = additional
            return primary
    return {}


@tool
def finish_tool(
    product_identified: bool = False,
    product_details: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    relevant_documents: Optional[Union[List[Any], Any]] = None,
    relevant_images: Optional[Union[List[Any], Any]] = None,
    past_tickets: Optional[Union[List[Any], Any]] = None,
    confidence: float = 0.5,
    reasoning: str = ""
) -> Dict[str, Any]:
    """
    Submit final gathered context and stop the ReACT loop.
    
    FIXED: More lenient with input types - accepts lists, dicts, or strings
    
    Use this tool when:
    - You have enough information to help the customer
    - You've searched all available sources
    - You're confident in the information gathered
    - You've reached iteration limit
    
    Args:
        product_identified: Did you successfully identify the product?
        product_details: Product info - can be a dict OR a list of dicts for multiple products
                        Example dict: {"model": "DKM.2420", "name": "Diverter Valve", "category": "Bathing"}
                        Example list: [{"model": "DKM.2420", ...}, {"model": "196.1280", ...}]
        relevant_documents: Documents found (flexible: list, dict, or string)
        relevant_images: Product images (flexible: list, dict, or string)
        past_tickets: Similar tickets (flexible: list, dict, or string)
        confidence: Your confidence (0.0-1.0)
        reasoning: What you found and any gaps
    
    Returns:
        {
            "finished": True,
            "summary": str,
            "context_quality": str
        }
    """
    logger.info(f"[FINISH] Product: {product_identified}, Confidence: {confidence:.2f}")
    
    # Normalize inputs - be VERY lenient
    # Handle product_details being either a dict OR a list
    normalized_product = _normalize_product_details(product_details)
    
    # Convert various input formats to lists
    docs = _safe_extract_list(relevant_documents)
    images = _safe_extract_list(relevant_images)
    tickets = _safe_extract_list(past_tickets)
    
    logger.info(f"[FINISH] Resources: {len(docs)} docs, {len(images)} images, {len(tickets)} tickets")
    
    # Assess context quality
    score = 0
    
    if product_identified:
        score += 30
    if len(docs) >= 3:
        score += 25
    elif len(docs) >= 1:
        score += 15
    if len(images) >= 1:
        score += 15
    if len(tickets) >= 2:
        score += 20
    elif len(tickets) >= 1:
        score += 10
    if confidence >= 0.8:
        score += 10
    
    if score >= 80:
        quality = "excellent"
    elif score >= 60:
        quality = "good"
    elif score >= 40:
        quality = "fair"
    else:
        quality = "poor"
    
    # Build summary
    parts = []
    
    if product_identified:
        model = normalized_product.get("model", "Unknown")
        name = normalized_product.get("name", "Product")
        parts.append(f"✅ Product: {model} ({name})")
        
        # Note additional products if present
        additional = normalized_product.get("additional_products", [])
        if additional:
            additional_models = [p.get("model", "?") for p in additional]
            parts.append(f"📦 Related: {', '.join(additional_models)}")
    else:
        parts.append("⚠️ Product not identified")
    
    parts.append(f"📄 Documents: {len(docs)}")
    parts.append(f"🖼️ Images: {len(images)}")
    parts.append(f"🎫 Past tickets: {len(tickets)}")
    parts.append(f"🎯 Confidence: {confidence:.0%}")
    
    if reasoning:
        parts.append(f"💭 {reasoning}")
    
    summary = " | ".join(parts)
    
    logger.info(f"[FINISH] Quality: {quality} ({score}/100)")
    logger.info(f"[FINISH] {summary}")
    
    return {
        "finished": True,
        "product_identified": product_identified,
        "product_details": normalized_product,
        "relevant_documents": docs,
        "relevant_images": images,
        "past_tickets": tickets,
        "confidence": confidence,
        "reasoning": reasoning,
        "summary": summary,
        "context_quality": quality,
        "context_score": score
    }