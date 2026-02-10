"""
Spare Parts Pricing Tool
LangChain tool for the React agent to look up spare part pricing.

Use this tool when:
- Customer asks for price of a specific spare part
- Dealer inquires about replacement part costs
- Need to quote spare part pricing for repairs
"""

from langchain.tools import tool
import logging
from typing import Dict, Any, Optional

from app.services.spare_parts_pricing_service import (
    find_spare_part_pricing,
    get_all_parts_for_base_model,
    normalize_part_number,
    extract_base_model
)

logger = logging.getLogger(__name__)


@tool
def spare_parts_pricing_tool(
    part_number: str,
    include_variants: bool = False
) -> Dict[str, Any]:
    """
    Look up spare part pricing from the Flusso spare parts database.
    
    Use this tool when you need to find the price of a spare part, replacement 
    component, or accessory. This database contains pricing for ~950 spare parts
    that are NOT in the main product catalog.
    
    WHEN TO USE:
    - Customer asks "How much is part TVH.5007?"
    - Dealer needs pricing for replacement parts
    - Quote needed for spare/replacement components
    - Part numbers starting with: TVL, TVH, TRM, RP, PBV, K., P., MEM, etc.
    
    PART NUMBER FORMATS RECOGNIZED:
    - "TVH.5007" → Category.BaseNumber
    - "100.1800-2353CP" → Series.BaseNumber-VariantFinish
    - "TRM.TVH.4511CP" → Prefix.Category.BaseNumberFinish
    - "RP70823" → SimpleCode
    
    Args:
        part_number: The spare part number to look up (e.g., "TVH.5007", "100.1800-2353CP")
        include_variants: If True, return all finish variants for the base model
        
    Returns:
        Dictionary with:
        - success: Whether parts were found
        - parts: List of matching parts with pricing
        - message: Human-readable result summary
        
    Price Status:
    - "available": Price is set (e.g., "$24.00")
    - "not_set": Part exists but price not configured (shows as "$ -")
    - Obsolete parts are flagged as is_obsolete=True
    """
    
    logger.info(f"[SPARE_PARTS_TOOL] Looking up: {part_number}")
    
    if not part_number or not part_number.strip():
        return {
            "success": False,
            "message": "Part number is required. Please provide a valid spare part number.",
            "parts": []
        }
    
    try:
        # Perform lookup
        result = find_spare_part_pricing(
            part_number=part_number.strip(),
            allow_fuzzy=True,
            limit=10
        )
        
        # If include_variants requested and we have a match, get all variants
        if include_variants and result.get("success"):
            parts = result.get("parts", [])
            if parts:
                # Get base model from first match
                first_part = parts[0].get("part_number", "")
                base_model = extract_base_model(normalize_part_number(first_part))
                variants = get_all_parts_for_base_model(base_model)
                if len(variants) > len(parts):
                    result["parts"] = variants
                    result["message"] += f" (showing all {len(variants)} finish variants)"
        
        # Format response for agent consumption
        if result.get("success"):
            parts = result.get("parts", [])
            
            # Build summary for agent
            summary_lines = []
            for p in parts[:5]:  # Limit display to 5
                part_num = p.get("part_number", "Unknown")
                price = p.get("price", "$ -")
                status = p.get("price_status", "unknown")
                
                if status == "not_set":
                    summary_lines.append(f"• {part_num}: Price not set (contact sales)")
                elif p.get("is_obsolete"):
                    summary_lines.append(f"• {part_num}: {price} (OBSOLETE)")
                elif p.get("is_display_dummy"):
                    summary_lines.append(f"• {part_num}: Display only, not for sale")
                else:
                    summary_lines.append(f"• {part_num}: {price}")
            
            if len(parts) > 5:
                summary_lines.append(f"  ... and {len(parts) - 5} more")
            
            result["summary"] = "\n".join(summary_lines)
            
            logger.info(f"[SPARE_PARTS_TOOL] ✅ Found {len(parts)} part(s)")
        else:
            logger.info(f"[SPARE_PARTS_TOOL] ❌ No match for {part_number}")
            
            # Add helpful suggestions
            suggestions = result.get("suggestions", [])
            if suggestions:
                result["message"] += f"\n\nDid you mean: {', '.join(suggestions[:3])}?"
        
        return result
        
    except Exception as e:
        logger.error(f"[SPARE_PARTS_TOOL] Error looking up {part_number}: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Error looking up spare part: {str(e)}",
            "parts": []
        }


@tool
def spare_parts_variants_tool(base_model: str) -> Dict[str, Any]:
    """
    Get all finish variants for a spare part base model.
    
    Use this when customer asks about available finishes for a spare part,
    or when you need to show all color/finish options with their prices.
    
    Args:
        base_model: The base model number without finish code (e.g., "100.1800-2353")
        
    Returns:
        Dictionary with all finish variants and their prices
    """
    
    logger.info(f"[SPARE_PARTS_VARIANTS] Looking up variants for: {base_model}")
    
    if not base_model or not base_model.strip():
        return {
            "success": False,
            "message": "Base model number is required.",
            "variants": []
        }
    
    try:
        variants = get_all_parts_for_base_model(base_model.strip())
        
        if variants:
            # Build summary
            summary_lines = [f"Available finishes for {base_model}:"]
            for v in variants:
                part_num = v.get("part_number", "")
                price = v.get("price", "$ -")
                has_price = v.get("has_price", False)
                
                # Extract finish code
                finish = part_num[-2:] if len(part_num) >= 2 else ""
                finish_name = _get_finish_name(finish)
                
                if has_price:
                    summary_lines.append(f"• {finish} ({finish_name}): {price}")
                else:
                    summary_lines.append(f"• {finish} ({finish_name}): Price not set")
            
            return {
                "success": True,
                "base_model": base_model,
                "variants": variants,
                "count": len(variants),
                "summary": "\n".join(summary_lines),
                "message": f"Found {len(variants)} finish variant(s)"
            }
        else:
            return {
                "success": False,
                "base_model": base_model,
                "variants": [],
                "message": f"No variants found for base model '{base_model}'"
            }
            
    except Exception as e:
        logger.error(f"[SPARE_PARTS_VARIANTS] Error: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Error looking up variants: {str(e)}",
            "variants": []
        }


def _get_finish_name(code: str) -> str:
    """Get human-readable finish name from code."""
    finish_map = {
        "CP": "Chrome",
        "BN": "Brushed Nickel",
        "PN": "Polished Nickel",
        "MB": "Matte Black",
        "SB": "Satin Brass",
        "BB": "Brushed Bronze",
        "SS": "Stainless Steel",
        "BG": "Brushed Gold",
        "PS": "Polished Steel",
        "GW": "Gloss White",
        "GB": "Gloss Black",
        "RB": "Rough Brass",
    }
    return finish_map.get(code.upper(), code)
