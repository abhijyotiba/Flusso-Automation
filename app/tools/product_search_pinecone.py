"""
Product Search Tool -
Metadata-first strategy for exact model number lookups
"""

import logging
import re
from typing import Dict, Any, Optional, List
from langchain.tools import tool

from app.clients.pinecone_client import get_pinecone_client
from app.clients.embeddings import embed_text_clip

logger = logging.getLogger(__name__)


def _looks_like_model_number(text: str) -> bool:
    """
    Heuristic: detect if a string looks like a model/part number.
    Examples: "100.1170", "160.1168-9862", "HS6270MB"
    """
    if not text:
        return False

    s = text.strip()
    if len(s) > 30:
        return False

    # Allow letters, digits, dot, dash, slash
    if not all(ch.isalnum() or ch in ".-/" for ch in s):
        return False

    # Require at least one digit
    if not any(ch.isdigit() for ch in s):
        return False

    return True


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
    - Customer mentions a specific model number (e.g., "HS6270MB", "F2580CP", "100.1170")
    - You need to verify a product exists in the catalog
    - You need product details (category, name, images)

    Args:
        query: Natural language description of the product OR a model/part string.
        model_number: Exact model number if explicitly known (e.g., "HS6270MB").
        category: Product category filter if known (e.g., "Shower Heads", "Kitchen Faucets").
        top_k: Number of results to return (default: 5)

    Returns:
        {
            "success": bool,
            "products": [...],
            "count": int,
            "search_method": "metadata_exact" | "semantic" | "none" | "error",
            "message": str
        }
    """

    if not query and not model_number:
        return {
            "success": False,
            "products": [],
            "count": 0,
            "search_method": "none",
            "message": "Either query or model_number must be provided"
        }

    # If no query but we have a model number, use it as query too
    if not query and model_number:
        query = model_number

    logger.info(f"[PRODUCT_SEARCH] Query: '{query}', Model: {model_number}, Category: {category}")

    try:
        client = get_pinecone_client()

        # ------------------------------------------------------------
        # 1) Decide which string to treat as "model" for metadata lookup
        # ------------------------------------------------------------
        normalized_model: Optional[str] = None

        if model_number:
            normalized_model = model_number.strip().upper()
        elif query and _looks_like_model_number(query):
            # If LLM only passed query="100.1170", treat it as a model number
            normalized_model = query.strip().upper()

        # ============================================================
        # STRATEGY 1: PURE METADATA LOOKUP (for exact model numbers)
        # Use dummy vector to avoid semantic interference
        # ============================================================
        if normalized_model:
            logger.info(f"[PRODUCT_SEARCH] Strategy: Pure metadata lookup for model '{normalized_model}'")

            # CLIP embeddings (ViT-B-32) are 512-dim in your setup
            dummy_vector = [0.0] * 512

            # In most schemas this is "model_no"; if needed you can extend this
            filter_dict: Dict[str, Any] = {"model_no": {"$eq": normalized_model}}
            if category:
                filter_dict["product_category"] = {"$eq": category}

            results = client.query_images(vector=dummy_vector, top_k=top_k, filter_dict=filter_dict)

            if results:
                products = _format_product_results(results)
                logger.info(
                    f"[PRODUCT_SEARCH] ✅ Found {len(products)} exact match(es) for model {normalized_model}"
                )
                return {
                    "success": True,
                    "products": products,
                    "count": len(products),
                    "search_method": "metadata_exact",
                    "message": f"Found {len(products)} exact match(es) for model {normalized_model}"
                }
            else:
                logger.warning(
                    f"[PRODUCT_SEARCH] No exact match for model '{normalized_model}', "
                    f"falling back to semantic search"
                )

        # ============================================================
        # STRATEGY 2: SEMANTIC SEARCH (fallback or when no model number)
        # ============================================================
        if not query:
            return {
                "success": False,
                "products": [],
                "count": 0,
                "search_method": "none",
                "message": "No query available for semantic search"
            }

        logger.info(f"[PRODUCT_SEARCH] Strategy: Semantic search")
        vector = embed_text_clip(query)

        filter_dict: Dict[str, Any] = {}
        if category:
            filter_dict["product_category"] = {"$eq": category}

        results = client.query_images(vector=vector, top_k=top_k, filter_dict=filter_dict)

        if results:
            products = _format_product_results(results)
            logger.info(f"[PRODUCT_SEARCH] ✅ Found {len(products)} semantic match(es)")
            return {
                "success": True,
                "products": products,
                "count": len(products),
                "search_method": "semantic",
                "message": f"Found {len(products)} semantically similar product(s)"
            }

        # No results at all
        logger.warning(f"[PRODUCT_SEARCH] ❌ No products found for query: '{query}'")
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
    """Format Pinecone results into clean product records."""
    products: List[Dict[str, Any]] = []

    for hit in results:
        metadata = hit.get("metadata", {}) or {}

        # Extract image URLs
        image_urls: List[str] = []
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
            "similarity_score": round(hit.get("score", 0) * 100),
            "raw_score": hit.get("score", 0),
        }

        products.append(product)

    return products
