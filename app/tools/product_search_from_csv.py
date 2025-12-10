from langchain.tools import tool
from typing import Optional, Dict, Any
import logging

# Logic from Step 1
from app.services.product_catalog_cache import find_products_by_model
# Fallback Tool (Ensure 'product_search_pinecone.py' exists!)
from app.tools.product_search_pinecone import product_search_tool as pinecone_search

logger = logging.getLogger(__name__)

def _looks_like_model_number(text: str) -> bool:
    if not text: return False
    return any(char.isdigit() for char in text) and len(text.split()) < 3

@tool
def product_search_tool(
    query: Optional[str] = None, 
    model_number: Optional[str] = None,
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Primary product lookup tool. 
    1. Checks CSV Catalog for exact OR partial model matches (e.g. "100.1170" -> "100.1170-PC").
    2. Falls back to Pinecone for fuzzy/semantic search if no CSV matches found.
    """
    
    # 1. Normalize Input
    target_model = None
    if model_number:
        target_model = model_number.strip().upper()
    elif query and _looks_like_model_number(query):
        target_model = query.strip().upper()

    # ==========================================
    # STRATEGY 1: CSV PREFIX SEARCH
    # ==========================================
    if target_model:
        logger.info(f"[PRODUCT_LOOKUP] Checking cache for: {target_model}")
        
        cached_matches = find_products_by_model(target_model, limit=10)
        
        if cached_matches:
            count = len(cached_matches)
            logger.info(f"[PRODUCT_LOOKUP] ✅ Found {count} match(es) in cache")
            
            msg = f"Found {count} match(es) in catalog."
            if count > 1:
                # Provide hints about variations to the Agent
                examples = ", ".join([p.get('model_no', 'N/A') for p in cached_matches[:5]])
                msg += f" Note: '{target_model}' appears to be a group number. Variations: {examples}"
                
            return {
                "success": True,
                "source": "catalog_cache",
                "products": cached_matches,
                "count": count,
                "message": msg
            }
        else:
             logger.info(f"[PRODUCT_LOOKUP] ❌ Not found in cache: {target_model}")

    # ==========================================
    # STRATEGY 2: PINECONE FALLBACK
    # ==========================================
    logger.info(f"[PRODUCT_LOOKUP] Falling back to Pinecone...")
    try:
        # Pass inputs to the backup tool
        pinecone_input = {}
        if query: pinecone_input["query"] = query
        if model_number: pinecone_input["model_number"] = model_number
        if category: pinecone_input["category"] = category
        
        # Use .run() to execute the LangChain tool
        return pinecone_search.run(tool_input=pinecone_input)
    except Exception as e:
        logger.error(f"[PRODUCT_LOOKUP] Pinecone fallback error: {e}")
        return {"success": False, "message": "Search failed in both systems."}