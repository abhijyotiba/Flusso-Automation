from langchain.tools import tool
from typing import Optional
import logging

from app.services.product_catalog_cache import get_product
from app.tools.product_search import product_search_tool  # Pinecone fallback

logger = logging.getLogger(__name__)


@tool
def product_lookup_tool(model_number: Optional[str] = None):
    """
    Primary product lookup tool using Google Sheet cache.
    Falls back to Pinecone product_search_tool if not found.
    """
    if not model_number:
        return {"success": False, "message": "model_number is required"}

    model = model_number.strip().upper()

    # ==========================================
    # STEP 1: Try in-memory Google Sheet lookup
    # ==========================================
    logger.info(f"[product_lookup_tool] Looking up model: {model}")

    product = get_product(model)

    if product:
        logger.info(f"[product_lookup_tool] Found product in sheet: {model}")
        return {
            "success": True,
            "source": "google_sheet",
            "product": product,
            "message": f"Found exact match for {model}"
        }

    logger.warning(f"[product_lookup_tool] Model {model} not found in sheet. Trying Pinecone...")

    # ==========================================
    # STEP 2: Pinecone fallback semantic search
    # ==========================================
    try:
        pinecone_result = product_search_tool.run(tool_input={"model_number": model})

        if pinecone_result.get("success") and pinecone_result.get("count", 0) > 0:
            return {
                "success": True,
                "source": "pinecone_fallback",
                "products": pinecone_result.get("products"),
                "message": f"No sheet match. Pinecone returned {pinecone_result.get('count')} results."
            }

        return {
            "success": False,
            "source": "none",
            "message": f"Model {model} not found in sheet or Pinecone"
        }

    except Exception as e:
        logger.error(f"[product_lookup_tool] Pinecone fallback failed: {e}")
        return {
            "success": False,
            "source": "error",
            "message": "Lookup failed in both sheet and Pinecone."
        }
