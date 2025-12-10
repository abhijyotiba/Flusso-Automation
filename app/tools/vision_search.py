"""
Vision Search Tool - CLIP Image Similarity
Identifies products from customer-provided images
"""

import logging
from typing import Dict, Any, List
from langchain.tools import tool

from app.clients.embeddings import embed_image
from app.clients.pinecone_client import get_pinecone_client
from app.config.settings import settings

logger = logging.getLogger(__name__)


@tool
def vision_search_tool(
    image_urls: List[str],
    expected_category: str = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Identify products from customer images using visual similarity.
    
    Use this tool when:
    - Customer attached product photos
    - No model number is mentioned
    - Need to identify product visually
    - Verify product appearance
    
    Args:
        image_urls: List of image URLs from ticket attachments
        expected_category: Expected product category from context 
                        (e.g., "Shower Heads", "Faucets")
        top_k: Number of similar products to return (default: 5)
    
    Returns:
        {
            "success": bool,
            "matches": [
                {
                    "model_no": str,
                    "product_title": str,
                    "category": str,
                    "similarity_score": float,
                    "image_url": str,
                    "confidence_level": "HIGH" | "MEDIUM" | "LOW"
                }
            ],
            "match_quality": "HIGH" | "MEDIUM" | "LOW" | "CATEGORY_MISMATCH" | "NO_MATCH",
            "reasoning": str,
            "count": int,
            "message": str
        }
    """
    logger.info(f"[VISION_SEARCH] Processing {len(image_urls)} image(s), Expected: {expected_category}")
    
    if not image_urls:
        return {
            "success": False,
            "matches": [],
            "match_quality": "NO_MATCH",
            "reasoning": "No images provided",
            "count": 0,
            "message": "No images to analyze"
        }
    
    try:
        client = get_pinecone_client()
        all_matches = []
        
        # Process each image
        for idx, img_url in enumerate(image_urls, 1):
            logger.info(f"[VISION_SEARCH] Processing image {idx}/{len(image_urls)}: {img_url}")
            
            try:
                # Generate CLIP embedding
                vector = embed_image(img_url)
                
                # Query Pinecone
                results = client.query_images(vector=vector, top_k=top_k)
                
                if results:
                    all_matches.extend(results)
                    logger.info(f"[VISION_SEARCH] Image {idx}: Found {len(results)} matches")
                    
            except Exception as e:
                logger.error(f"[VISION_SEARCH] Failed to process image {idx}: {e}")
                continue
        
        if not all_matches:
            return {
                "success": False,
                "matches": [],
                "match_quality": "NO_MATCH",
                "reasoning": "No visually similar products found in catalog",
                "count": 0,
                "message": "Could not find matching products"
            }
        
        # Sort by similarity score and deduplicate
        all_matches.sort(key=lambda x: x.get("score", 0), reverse=True)
        unique_matches = _deduplicate_matches(all_matches, top_k)
        
        # Assess match quality
        top_score = unique_matches[0].get("score", 0)
        top_category = unique_matches[0].get("metadata", {}).get("product_category", "Unknown")
        
        match_quality, reasoning = _assess_match_quality(
            top_score=top_score,
            top_category=top_category,
            expected_category=expected_category,
            threshold=settings.vision_min_similarity_threshold
        )
        
        # Format results
        formatted_matches = []
        for match in unique_matches:
            metadata = match.get("metadata", {})
            score = match.get("score", 0)
            
            formatted_matches.append({
                "model_no": metadata.get("model_no", "N/A"),
                "product_title": metadata.get("product_title", "Unknown"),
                "category": metadata.get("product_category", "Unknown"),
                "sub_category": metadata.get("sub_category", ""),
                "finish": metadata.get("finish", "N/A"),
                "similarity_score": round(score * 100),
                "image_url": metadata.get("image_url", ""),
                "confidence_level": _score_to_confidence(score)
            })
        
        logger.info(f"[VISION_SEARCH] Match quality: {match_quality}, Top score: {top_score:.3f}")
        
        # Fix: If category mismatch, mark as unsuccessful and don't return bad matches (improvement from improvements.md)
        if match_quality == "CATEGORY_MISMATCH":
            logger.warning(f"[VISION_SEARCH] Category mismatch detected - rejecting results")
            return {
                "success": False,  # Mark as unsuccessful
                "matches": [],      # Don't return bad matches
                "match_quality": "CATEGORY_MISMATCH",
                "reasoning": reasoning,
                "count": 0,
                "message": f"Vision search found products from different category - results excluded"
            }
        
        return {
            "success": True,
            "matches": formatted_matches,
            "match_quality": match_quality,
            "reasoning": reasoning,
            "count": len(formatted_matches),
            "message": f"Found {len(formatted_matches)} visually similar product(s)"
        }
        
    except Exception as e:
        logger.error(f"[VISION_SEARCH] Error: {e}", exc_info=True)
        return {
            "success": False,
            "matches": [],
            "match_quality": "NO_MATCH",
            "reasoning": f"Search error: {str(e)}",
            "count": 0,
            "message": f"Vision search failed: {str(e)}"
        }


def _deduplicate_matches(matches: List[Dict], top_k: int) -> List[Dict]:
    """Remove duplicate products based on model number"""
    seen_models = set()
    unique = []
    
    for match in matches:
        model = match.get("metadata", {}).get("model_no")
        if model and model not in seen_models:
            seen_models.add(model)
            unique.append(match)
            if len(unique) >= top_k:
                break
    
    return unique


def _assess_match_quality(
    top_score: float,
    top_category: str,
    expected_category: str,
    threshold: float
) -> tuple:
    """Assess the quality of vision matches"""
    
    # Check if below threshold
    if top_score < threshold:
        return "LOW", f"Top similarity score ({top_score:.2f}) is below threshold ({threshold})"
    
    # Check category mismatch
    if expected_category:
        expected_lower = expected_category.lower()
        top_lower = top_category.lower()
        
        # Strict category matching
        if expected_lower not in top_lower and top_lower not in expected_lower:
            return "CATEGORY_MISMATCH", f"Expected '{expected_category}' but found '{top_category}'"
    
    # Assess confidence based on score
    if top_score >= 0.85:
        return "HIGH", f"Strong visual match with {top_score:.2%} similarity"
    elif top_score >= 0.70:
        return "MEDIUM", f"Moderate visual match with {top_score:.2%} similarity"
    else:
        return "LOW", f"Weak visual match with {top_score:.2%} similarity"


def _score_to_confidence(score: float) -> str:
    """Convert similarity score to confidence level"""
    if score >= 0.85:
        return "HIGH"
    elif score >= 0.70:
        return "MEDIUM"
    else:
        return "LOW"
