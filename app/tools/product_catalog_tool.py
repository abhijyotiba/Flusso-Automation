"""
Product Catalog Tool - Comprehensive Product Search for Flusso Products

This tool provides intelligent product search capabilities with multiple strategies:
1. Exact model number lookup (instant)
2. Group/prefix matching for variations (instant)
3. Fuzzy matching for typos (fast)
4. Keyword-based text search (fast)
5. Category and collection filtering

Data: 5,687 products with 70 fields each, including:
- Full product details (title, description, features)
- Pricing (list price, MAP price)
- Specifications (dimensions, flow rate, etc.)
- Document links (spec sheets, install manuals, parts diagrams)
- Finish variations and availability
- Related spare parts

Use this tool to:
- Find products by model number
- Verify extracted model numbers from images/documents
- Get product specifications and pricing
- Find available finish variations
- Search by product description or keywords
- Filter by category or collection
"""

import logging
from typing import Dict, Any, Optional, List
from langchain.tools import tool

from app.services.product_catalog import (
    ensure_catalog_loaded,
    looks_like_model_number,
    get_finish_name,
    FINISH_CODE_MAP,
)

logger = logging.getLogger(__name__)


def _format_product_summary(product: Dict[str, Any], include_details: bool = True) -> Dict[str, Any]:
    """
    Format a product for output with essential fields.
    
    Args:
        product: Raw product dict from catalog
        include_details: Whether to include full details or just essentials
        
    Returns:
        Formatted product dict
    """
    # Always include core fields
    formatted = {
        "model_no": product["model_no"],
        "group_number": product["group_number"],
        "title": product["title"],
        "category": product["category"],
        "sub_category": product["sub_category"],
        "collection": product["collection"],
        "finish_code": product["finish_code"],
        "finish_name": product["finish_name"],
        "list_price": product["list_price"],
        "is_active": product["is_active"],
        "is_spare_part": product["is_spare_part"],
    }
    
    if include_details:
        # Add detailed fields
        formatted.update({
            "description": product["description"][:500] if product["description"] else "",
            "features": product["features"],
            "map_price": product["map_price"],
            "flow_rate_gpm": product["flow_rate_gpm"],
            "dimensions": {
                "height": product["height_inches"],
                "length": product["length_inches"],
                "width": product["width_inches"],
            },
            "weight_lbs": product["weight_lbs"],
            "holes_needed": product["holes_needed"],
            "is_touch_capable": product["is_touch_capable"],
            "warranty": product["warranty"],
            "product_url": product["product_url"],
            "image_url": product["image_url"],
            "spec_sheet_url": product["spec_sheet_url"],
            "install_manual_url": product["install_manual_url"],
            "parts_diagram_url": product["parts_diagram_url"],
            "install_video_url": product["install_video_url"],
        })
    
    return formatted


@tool
def product_catalog_tool(
    query: Optional[str] = None,
    model_number: Optional[str] = None,
    category: Optional[str] = None,
    collection: Optional[str] = None,
    include_variations: bool = True,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Search the Flusso product catalog (5,687 products with full details).
    
    This is your primary tool for finding product information. It has:
    - COMPLETE product data (70 fields per product)
    - ALL 5,687 products including 783 product groups
    - Finish variations (29 different finishes like Chrome, Matte Black, etc.)
    - Pricing, specifications, document links, and more
    
    SEARCH CAPABILITIES:
    
    1. **Exact Model Search** - Use when you have a specific model number
       Example: model_number="100.1170CP" → Returns exact product
       
    2. **Group/Base Model Search** - Finds all finish variations
       Example: model_number="100.1170" → Returns all finishes (CP, BN, MB, etc.)
       
    3. **Fuzzy Search** - Handles typos in model numbers
       Example: model_number="100.117" → Suggests "100.1170"
       
    4. **Keyword Search** - Search by description
       Example: query="floor mount tub faucet chrome"
       
    5. **Category Filter** - Filter by product category
       Categories: Showering, Bathing, Sink Faucets, Kitchen, Bath Accessories, Spare Parts
       
    6. **Collection Filter** - Filter by collection
       Collections: Serie 100, Serie 196, Universal Fixtures, Cascade, etc.
    
    PRODUCT DATA INCLUDES:
    - Identification: model_no, group_number, UPC
    - Details: title, description, features (bullet points)
    - Classification: category, sub_category, collection, style
    - Finish: finish_code (CP, BN, MB), finish_name (Chrome, Brushed Nickel)
    - Pricing: list_price, map_price
    - Specs: dimensions, weight, flow_rate_gpm, holes_needed
    - Documents: spec_sheet_url, install_manual_url, parts_diagram_url
    - Videos: install_video_url, operational_video_url
    - Status: is_active, is_spare_part, warranty
    
    FINISH CODES REFERENCE:
    CP=Chrome, BN=Brushed Nickel, PN=Polished Nickel, MB=Matte Black,
    SB=Satin Brass, BB=Brushed Bronze, GW=Gloss White, GB=Gloss Black,
    SS=Stainless Steel, RB=Rough Brass
    
    Args:
        query: Natural language search (e.g., "floor mount tub faucet")
        model_number: Specific model or group number (e.g., "100.1170CP" or "100.1170")
        category: Filter by category (Showering, Bathing, Sink Faucets, Kitchen, etc.)
        collection: Filter by collection (Serie 100, Serie 196, etc.)
        include_variations: If True and searching by group, include all finish variations
        limit: Maximum number of results to return (default: 10)
    
    Returns:
        {
            "success": bool,
            "search_method": "exact" | "group" | "prefix" | "fuzzy" | "keyword" | "category",
            "query_interpreted": str,
            "products": [...],
            "count": int,
            "variations": {...} (if group search),
            "suggestions": [...] (if fuzzy match),
            "related_parts": [...] (if exact match found),
            "message": str
        }
    """
    logger.info(f"[PRODUCT_CATALOG] Search: query={query}, model={model_number}, "
               f"cat={category}, collection={collection}")
    
    # Ensure catalog is loaded
    try:
        catalog = ensure_catalog_loaded()
    except Exception as e:
        logger.error(f"[PRODUCT_CATALOG] Failed to load catalog: {e}")
        return {
            "success": False,
            "search_method": "error",
            "products": [],
            "count": 0,
            "message": f"Failed to load product catalog: {str(e)}"
        }
    
    if not catalog.products:
        return {
            "success": False,
            "search_method": "error",
            "products": [],
            "count": 0,
            "message": "Product catalog is empty. Check if metadata_manifest.json exists."
        }
    
    # Determine search strategy
    search_target = None
    search_method = "none"
    
    # Priority 1: Explicit model number
    if model_number:
        search_target = model_number.strip().upper()
        search_method = "model"
    # Priority 2: Query that looks like a model number
    elif query and looks_like_model_number(query):
        search_target = query.strip().upper()
        search_method = "model"
    # Priority 3: Category-only search
    elif category and not query:
        search_method = "category"
    # Priority 4: Collection-only search
    elif collection and not query:
        search_method = "collection"
    # Priority 5: Keyword search
    elif query:
        search_method = "keyword"
    else:
        return {
            "success": False,
            "search_method": "none",
            "products": [],
            "count": 0,
            "message": "Please provide a query, model_number, category, or collection to search."
        }
    
    # ==========================================================================
    # STRATEGY 1: MODEL NUMBER SEARCH (exact, group, prefix, fuzzy)
    # ==========================================================================
    if search_method == "model":
        logger.info(f"[PRODUCT_CATALOG] Model search for: {search_target}")
        
        # Try exact match first
        exact_product = catalog.search_exact_model(search_target)
        if exact_product:
            logger.info(f"[PRODUCT_CATALOG] ✅ Exact match found: {search_target}")
            
            products = [_format_product_summary(exact_product, include_details=True)]
            
            # Get variations if requested
            variations = {}
            if include_variations:
                variations = catalog.get_finish_variations(exact_product["group_number"])
            
            # Get related parts
            related_parts = catalog.get_related_parts(search_target)
            related_formatted = [_format_product_summary(p, include_details=False) for p in related_parts]
            
            return {
                "success": True,
                "search_method": "exact",
                "query_interpreted": search_target,
                "products": products,
                "count": 1,
                "variations": variations,
                "related_parts": related_formatted,
                "message": f"Found exact match for model {search_target}"
            }
        
        # Try group match (base model without finish suffix)
        group_products = catalog.search_by_group(search_target)
        if group_products:
            logger.info(f"[PRODUCT_CATALOG] ✅ Group match found: {len(group_products)} variations")
            
            products = [_format_product_summary(p, include_details=True) for p in group_products[:limit]]
            variations = {}
            for p in group_products:
                if p["finish_code"]:
                    variations[p["finish_code"]] = p["finish_name"]
            
            return {
                "success": True,
                "search_method": "group",
                "query_interpreted": search_target,
                "products": products,
                "count": len(group_products),
                "variations": variations,
                "message": f"Found {len(group_products)} finish variation(s) for group {search_target}"
            }
        
        # Try prefix match
        prefix_products = catalog.search_prefix(search_target, limit=limit)
        if prefix_products:
            logger.info(f"[PRODUCT_CATALOG] ✅ Prefix match found: {len(prefix_products)} products")
            
            products = [_format_product_summary(p, include_details=True) for p in prefix_products]
            
            return {
                "success": True,
                "search_method": "prefix",
                "query_interpreted": search_target,
                "products": products,
                "count": len(prefix_products),
                "message": f"Found {len(prefix_products)} product(s) starting with {search_target}"
            }
        
        # Try fuzzy match (typo correction)
        fuzzy_results = catalog.search_fuzzy(search_target, limit=limit)
        if fuzzy_results:
            logger.info(f"[PRODUCT_CATALOG] ✅ Fuzzy match found: {len(fuzzy_results)} similar")
            
            products = []
            suggestions = []
            for product, score in fuzzy_results:
                products.append(_format_product_summary(product, include_details=True))
                suggestions.append({
                    "model_no": product["model_no"],
                    "similarity": round(score * 100),
                    "title": product["title"]
                })
            
            return {
                "success": True,
                "search_method": "fuzzy",
                "query_interpreted": search_target,
                "products": products,
                "count": len(products),
                "suggestions": suggestions,
                "message": f"No exact match for '{search_target}'. Did you mean: {suggestions[0]['model_no']}?"
            }
        
        # No match found
        logger.warning(f"[PRODUCT_CATALOG] ❌ No match found for: {search_target}")
        return {
            "success": False,
            "search_method": "model",
            "query_interpreted": search_target,
            "products": [],
            "count": 0,
            "message": f"No products found matching model '{search_target}'. Check the model number format."
        }
    
    # ==========================================================================
    # STRATEGY 2: CATEGORY SEARCH
    # ==========================================================================
    if search_method == "category":
        logger.info(f"[PRODUCT_CATALOG] Category search: {category}")
        
        products_raw = catalog.search_by_category(category, limit=limit)
        
        if products_raw:
            products = [_format_product_summary(p, include_details=False) for p in products_raw]
            return {
                "success": True,
                "search_method": "category",
                "query_interpreted": category,
                "products": products,
                "count": len(catalog.category_index.get(category.upper(), [])),
                "message": f"Found {len(products)} product(s) in category '{category}' (showing first {limit})"
            }
        else:
            available = ", ".join(catalog.get_categories()[:10])
            return {
                "success": False,
                "search_method": "category",
                "query_interpreted": category,
                "products": [],
                "count": 0,
                "message": f"Category '{category}' not found. Available: {available}"
            }
    
    # ==========================================================================
    # STRATEGY 3: COLLECTION SEARCH
    # ==========================================================================
    if search_method == "collection":
        logger.info(f"[PRODUCT_CATALOG] Collection search: {collection}")
        
        products_raw = catalog.search_by_collection(collection, limit=limit)
        
        if products_raw:
            products = [_format_product_summary(p, include_details=False) for p in products_raw]
            return {
                "success": True,
                "search_method": "collection",
                "query_interpreted": collection,
                "products": products,
                "count": len(catalog.collection_index.get(collection.upper(), [])),
                "message": f"Found {len(products)} product(s) in collection '{collection}' (showing first {limit})"
            }
        else:
            available = ", ".join(catalog.get_collections()[:10])
            return {
                "success": False,
                "search_method": "collection",
                "query_interpreted": collection,
                "products": [],
                "count": 0,
                "message": f"Collection '{collection}' not found. Available: {available}"
            }
    
    # ==========================================================================
    # STRATEGY 4: KEYWORD SEARCH
    # ==========================================================================
    if search_method == "keyword":
        logger.info(f"[PRODUCT_CATALOG] Keyword search: {query}")
        
        products_raw = catalog.search_keywords(query, category=category, 
                                               collection=collection, limit=limit)
        
        if products_raw:
            products = [_format_product_summary(p, include_details=True) for p in products_raw]
            
            filter_msg = ""
            if category:
                filter_msg += f" in category '{category}'"
            if collection:
                filter_msg += f" in collection '{collection}'"
            
            return {
                "success": True,
                "search_method": "keyword",
                "query_interpreted": query,
                "products": products,
                "count": len(products),
                "message": f"Found {len(products)} product(s) matching '{query}'{filter_msg}"
            }
        else:
            return {
                "success": False,
                "search_method": "keyword",
                "query_interpreted": query,
                "products": [],
                "count": 0,
                "message": f"No products found matching '{query}'. Try different keywords or a model number."
            }
    
    # Fallback (should not reach here)
    return {
        "success": False,
        "search_method": "unknown",
        "products": [],
        "count": 0,
        "message": "Unable to determine search strategy."
    }


@tool
def get_product_variations(group_number: str) -> Dict[str, Any]:
    """
    Get all available finish variations for a product group.
    
    Use this when you need to show a customer all available finishes for a product.
    
    Args:
        group_number: The base/group model number (e.g., "100.1170" without finish suffix)
    
    Returns:
        {
            "success": bool,
            "group_number": str,
            "variations": {
                "CP": {"model_no": "100.1170CP", "finish_name": "Chrome", "price": 2150.0},
                "BN": {...},
                ...
            },
            "count": int,
            "message": str
        }
    """
    logger.info(f"[PRODUCT_CATALOG] Get variations for: {group_number}")
    
    try:
        catalog = ensure_catalog_loaded()
    except Exception as e:
        return {"success": False, "message": str(e)}
    
    normalized = group_number.strip().upper()
    products = catalog.search_by_group(normalized)
    
    if not products:
        return {
            "success": False,
            "group_number": normalized,
            "variations": {},
            "count": 0,
            "message": f"No products found for group '{normalized}'"
        }
    
    variations = {}
    for p in products:
        code = p["finish_code"] or "BASE"
        variations[code] = {
            "model_no": p["model_no"],
            "finish_name": p["finish_name"],
            "price": p["list_price"],
            "is_active": p["is_active"],
            "image_url": p["image_url"]
        }
    
    return {
        "success": True,
        "group_number": normalized,
        "variations": variations,
        "count": len(variations),
        "message": f"Found {len(variations)} finish variation(s) for {normalized}"
    }


@tool
def get_catalog_info() -> Dict[str, Any]:
    """
    Get information about the product catalog.
    
    Use this to understand what data is available in the catalog.
    
    Returns:
        Catalog statistics including total products, categories, collections, etc.
    """
    try:
        catalog = ensure_catalog_loaded()
    except Exception as e:
        return {"success": False, "message": str(e)}
    
    return {
        "success": True,
        "stats": catalog.get_stats(),
        "categories": catalog.get_categories(),
        "collections": catalog.get_collections(),
        "finish_codes": FINISH_CODE_MAP,
        "message": f"Catalog loaded with {catalog.stats['total_products']} products"
    }
