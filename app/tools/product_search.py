"""
Product Search Tool - Pinecone Metadata Query
Searches product catalog by model number or description
"""

import logging
from typing import Dict, Any, Optional, List
from langchain.tools import tool

from app.clients.pinecone_client import get_pinecone_client
from app.clients.embeddings import embed_text_clip

logger = logging.getLogger(__name__)


@tool
def product_search_tool(
    query: Optional[str] = None,
    model_number: Optional[str] = None,
    category: Optional[str] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Search product catalog by model number or description using Pinecone.
    
    Use this tool when:
    - Customer mentions a specific model number (e.g., "HS6270MB", "F2580CP")
    - You need to verify a product exists in the catalog
    - You need product details (category, name, images)
    
    Args:
        query: Natural language description of the product (optional if model_number provided)
        model_number: Exact model number if mentioned in ticket (e.g., "HS6270MB")
        category: Product category filter if known (e.g., "Shower Heads", "Kitchen Faucets")
        top_k: Number of results to return (default: 5)
    
    Returns:
        {
            "success": bool,
            "products": [
                {
                    "model_no": str,
                    "product_title": str,
                    "category": str,
                    "sub_category": str,
                    "finish": str,
                    "collection": str,
                    "image_urls": [str],
                    "similarity_score": float
                }
            ],
            "count": int,
            "search_method": "model_number" | "semantic" | "metadata",
            "message": str
        }
    """
    # Handle case where neither query nor model_number provided
    if not query and not model_number:
        return {
            "success": False,
            "products": [],
            "count": 0,
            "search_method": "none",
            "message": "Either query or model_number must be provided"
        }
    
    # Default query to model_number if not provided
    if not query and model_number:
        query = model_number
    
    logger.info(f"[PRODUCT_SEARCH] Query: '{query}', Model: {model_number}, Category: {category}")
    
    try:
        client = get_pinecone_client()
        
        # Strategy 1: Direct model number metadata filter (highest accuracy)
        if model_number:
            logger.info(f"[PRODUCT_SEARCH] Strategy: Direct model number lookup")
            # Use text embedding but filter by exact model
            vector = embed_text_clip(query)
            
            filter_dict = {"model_no": {"$eq": model_number.upper()}}
            if category:
                filter_dict["product_category"] = {"$eq": category}
            
            results = client.query_images(vector=vector, top_k=top_k, filter_dict=filter_dict)
            
            if results:
                products = _format_product_results(results)
                return {
                    "success": True,
                    "products": products,
                    "count": len(products),
                    "search_method": "model_number",
                    "message": f"Found {len(products)} exact match(es) for model {model_number}"
                }
        
        # Strategy 2: Semantic search with optional category filter
        logger.info(f"[PRODUCT_SEARCH] Strategy: Semantic search")
        vector = embed_text_clip(query)
        
        filter_dict = {}
        if category:
            filter_dict["product_category"] = {"$eq": category}
        
        results = client.query_images(vector=vector, top_k=top_k, filter_dict=filter_dict)
        
        if results:
            products = _format_product_results(results)
            return {
                "success": True,
                "products": products,
                "count": len(products),
                "search_method": "semantic",
                "message": f"Found {len(products)} semantically similar product(s)"
            }
        
        # No results
        return {
            "success": False,
            "products": [],
            "count": 0,
            "search_method": "none",
            "message": "No products found matching the criteria"
        }
        
    except Exception as e:
        logger.error(f"[PRODUCT_SEARCH] Error: {e}", exc_info=True)
        return {
            "success": False,
            "products": [],
            "count": 0,
            "search_method": "error",
            "message": f"Search failed: {str(e)}"
        }


def _format_product_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format Pinecone results into clean product records"""
    products = []
    
    for hit in results:
        metadata = hit.get("metadata", {})
        
        # Extract image URLs (may be in different metadata fields)
        image_urls = []
        if "image_url" in metadata:
            image_urls.append(metadata["image_url"])
        if "additional_images" in metadata and isinstance(metadata["additional_images"], list):
            image_urls.extend(metadata["additional_images"])
        
        product = {
            "model_no": metadata.get("model_no", metadata.get("product_id", "N/A")),
            "product_title": metadata.get("product_title", metadata.get("product_name", "Unknown")),
            "category": metadata.get("product_category", metadata.get("category", "Unknown")),
            "sub_category": metadata.get("sub_category", ""),
            "finish": metadata.get("finish", "N/A"),
            "collection": metadata.get("collection", ""),
            "image_urls": image_urls,
            "similarity_score": round(hit.get("score", 0) * 100),  # Convert to percentage
            "raw_score": hit.get("score", 0)
        }
        
        products.append(product)
    
    return products
