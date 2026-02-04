"""
Requirements Matrix - Defines what fields are required per ticket category.

This module defines the business rules for what information is REQUIRED
for each ticket category. The constraint_validator uses this to compute
missing_fields and must_not_ask lists.

Edit this file to add/modify requirements for ticket types.
"""

from typing import Dict, List, Any


# =============================================================================
# FIELD DEFINITIONS
# =============================================================================

# Map field keys (from ticket_facts) to human-readable names for customer asks
FIELD_NAMES: Dict[str, str] = {
    "receipt": "proof of purchase (receipt, invoice, or order confirmation)",
    "address": "shipping address for replacement delivery",
    "photos": "photo(s) showing the issue or defect",
    "video": "video showing the issue (especially helpful for intermittent problems)",
    "po": "PO number or order number",
    "model": "product model number",
    "finish": "product finish/color preference",
    "part_number": "specific part number needed",
}

# Map field keys to the ticket_facts boolean fields
FIELD_TO_FACTS_KEY: Dict[str, str] = {
    "receipt": "has_receipt",
    "address": "has_address",
    "photos": "has_photos",
    "video": "has_video",
    "po": "has_po",
    "model": "has_model_number",
    "finish": "raw_finish_mentions",  # Check if list is non-empty
    "part_number": "raw_part_numbers",  # Check if list is non-empty
}

# Customer-friendly request templates for missing fields
FIELD_ASK_TEMPLATES: Dict[str, str] = {
    "receipt": "Could you please provide your proof of purchase (receipt, invoice, or order confirmation)? This helps us verify your warranty coverage.",
    "address": "What address should we send the replacement to?",
    "photos": "Could you please send a photo showing the issue with your product? This helps us assess the problem accurately.",
    "video": "If possible, could you send a short video showing the issue? This is especially helpful for intermittent problems.",
    "po": "Could you please provide your PO number or order confirmation number?",
    "model": "Could you please provide the product model number? You can usually find this on the product label or in your order confirmation.",
    "finish": "What finish/color would you prefer for the replacement? (e.g., Chrome, Matte Black, Brushed Nickel)",
    "part_number": "Could you please specify which part(s) you need? If you're unsure, a photo of the product or parts diagram would help.",
}


# =============================================================================
# REQUIREMENTS MATRIX
# =============================================================================

# Requirements per ticket category
# Format:
#   "category_name": {
#       "required": [...],              # Always required fields
#       "conditional": {...},           # Required under certain conditions
#       "policies": [...],              # Policy rules to apply (from policy_rules.py)
#       "product_specific_policies": {} # Product-based policy overrides
#   }

REQUIREMENTS_MATRIX: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # WARRANTY / DEFECT CLAIMS
    # =========================================================================
    "warranty_claim": {
        "required": ["receipt", "address"],
        "conditional": {
            "photos": {
                "condition": "always_for_defect",
                "reason": "To verify and document the defect"
            },
            "video": {
                "condition": "intermittent_issue",
                "reason": "To show the intermittent problem occurring"
            }
        },
        "policies": ["warranty_standard"],
        "product_specific_policies": {
            "hose": "hose_warranty",
            "supply_line": "hose_warranty",
            "supply line": "hose_warranty",
            "braided": "hose_warranty",
        },
        "description": "Customer claiming warranty for defective product"
    },
    
    "product_issue": {
        "required": ["model"],
        "conditional": {
            "photos": {
                "condition": "always_for_defect",
                "reason": "To identify and document the issue"
            },
            "receipt": {
                "condition": "warranty_check_needed",
                "reason": "To verify warranty status if replacement needed"
            },
            "address": {
                "condition": "replacement_offered",
                "reason": "For shipping replacement parts"
            }
        },
        "policies": ["warranty_standard"],
        "product_specific_policies": {
            "hose": "hose_warranty",
            "supply_line": "hose_warranty",
        },
        "description": "Customer reporting product defect or malfunction"
    },
    
    # =========================================================================
    # MISSING PARTS
    # =========================================================================
    "missing_parts": {
        "required": ["po", "address"],
        "conditional": {
            "photos": {
                "condition": "unclear_what_missing",
                "reason": "To identify exactly which parts are missing"
            }
        },
        "policies": ["missing_parts_window"],
        "product_specific_policies": {},
        "description": "Customer reporting missing parts from order"
    },
    
    # =========================================================================
    # REPLACEMENT PARTS
    # =========================================================================
    "replacement_parts": {
        "required": ["model", "address"],
        "conditional": {
            "receipt": {
                "condition": "warranty_replacement",
                "reason": "To verify warranty coverage for free replacement"
            },
            "part_number": {
                "condition": "specific_part_needed",
                "reason": "To identify the exact part required"
            },
            "photos": {
                "condition": "part_identification_needed",
                "reason": "To identify the correct replacement part"
            }
        },
        "policies": ["warranty_standard"],
        "product_specific_policies": {},
        "description": "Customer requesting replacement parts"
    },
    
    # =========================================================================
    # RETURNS AND REFUNDS
    # =========================================================================
    "return_refund": {
        "required": ["receipt", "address"],
        "conditional": {
            "photos": {
                "condition": "damaged_product",
                "reason": "To document the product condition"
            }
        },
        "policies": ["return_policy"],
        "product_specific_policies": {},
        "description": "Customer requesting return or refund"
    },
    
    # =========================================================================
    # PRODUCT INQUIRIES
    # =========================================================================
    "product_inquiry": {
        "required": ["model"],
        "conditional": {
            "finish": {
                "condition": "finish_specific_question",
                "reason": "To identify the exact product variant"
            }
        },
        "policies": [],
        "product_specific_policies": {},
        "description": "Customer asking about product specs or compatibility"
    },
    
    "finish_color": {
        "required": ["model"],
        "conditional": {},
        "policies": [],
        "product_specific_policies": {},
        "description": "Customer asking about finish/color options"
    },
    
    # =========================================================================
    # INSTALLATION HELP
    # =========================================================================
    "installation_help": {
        "required": ["model"],
        "conditional": {
            "photos": {
                "condition": "installation_problem",
                "reason": "To see the current installation setup"
            }
        },
        "policies": [],
        "product_specific_policies": {},
        "description": "Customer needing installation assistance"
    },
    
    # =========================================================================
    # INFORMATION REQUESTS (NO PRODUCT ID NEEDED)
    # =========================================================================
    "pricing_request": {
        "required": [],
        "conditional": {
            "model": {
                "condition": "specific_product_pricing",
                "reason": "To provide accurate pricing for the specific product"
            },
            "part_number": {
                "condition": "part_pricing",
                "reason": "To look up the part price"
            }
        },
        "policies": [],
        "product_specific_policies": {},
        "description": "Customer asking about pricing"
    },
    
    "dealer_inquiry": {
        "required": [],
        "conditional": {},
        "policies": ["dealer_program"],
        "product_specific_policies": {},
        "description": "Dealer/partnership inquiry"
    },
    
    # =========================================================================
    # SPECIAL HANDLING
    # =========================================================================
    "shipping_tracking": {
        "required": ["po"],
        "conditional": {},
        "policies": [],
        "product_specific_policies": {},
        "description": "Customer asking about order status"
    },
    
    "feedback_suggestion": {
        "required": [],
        "conditional": {},
        "policies": [],
        "product_specific_policies": {},
        "description": "Customer providing feedback or suggestions"
    },
    
    # =========================================================================
    # DEFAULT / GENERAL
    # =========================================================================
    "general": {
        "required": [],
        "conditional": {},
        "policies": [],
        "product_specific_policies": {},
        "description": "General inquiry not fitting other categories"
    },
}


# =============================================================================
# CATEGORY ALIASES
# =============================================================================

# Maps various category names/keywords to canonical category names
CATEGORY_ALIASES: Dict[str, str] = {
    # Warranty variations
    "warranty": "warranty_claim",
    "defect": "warranty_claim",
    "defective": "warranty_claim",
    "broken": "product_issue",
    "malfunction": "product_issue",
    "not_working": "product_issue",
    "leaking": "product_issue",
    "leak": "product_issue",
    
    # Missing parts variations
    "missing": "missing_parts",
    "incomplete": "missing_parts",
    "not_included": "missing_parts",
    "parts_missing": "missing_parts",
    
    # Return variations
    "return": "return_refund",
    "refund": "return_refund",
    "rga": "return_refund",
    "send_back": "return_refund",
    
    # Replacement variations
    "replacement": "replacement_parts",
    "spare_part": "replacement_parts",
    "spare_parts": "replacement_parts",
    "parts": "replacement_parts",
    "need_part": "replacement_parts",
    
    # Inquiry variations
    "question": "product_inquiry",
    "inquiry": "product_inquiry",
    "compatibility": "product_inquiry",
    "spec": "product_inquiry",
    "specs": "product_inquiry",
    
    # Installation variations
    "install": "installation_help",
    "installation": "installation_help",
    "setup": "installation_help",
    "mounting": "installation_help",
    "how_to": "installation_help",
    
    # Finish variations
    "finish": "finish_color",
    "color": "finish_color",
    "colour": "finish_color",
    
    # Pricing variations
    "pricing": "pricing_request",
    "price": "pricing_request",
    "msrp": "pricing_request",
    "cost": "pricing_request",
    
    # Dealer variations
    "dealer": "dealer_inquiry",
    "partnership": "dealer_inquiry",
    "distributor": "dealer_inquiry",
    "wholesale": "dealer_inquiry",
    "account": "dealer_inquiry",
    
    # Shipping variations
    "shipping": "shipping_tracking",
    "tracking": "shipping_tracking",
    "delivery": "shipping_tracking",
    "order_status": "shipping_tracking",
    "where_is": "shipping_tracking",
    
    # Feedback variations
    "feedback": "feedback_suggestion",
    "suggestion": "feedback_suggestion",
    "complaint": "feedback_suggestion",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_canonical_category(category: str) -> str:
    """
    Convert a category string to its canonical form.
    
    Args:
        category: Category string (may be alias or canonical)
        
    Returns:
        Canonical category name
    """
    if not category:
        return "general"
    
    # Normalize: lowercase, replace spaces/hyphens with underscores
    normalized = category.lower().strip().replace(" ", "_").replace("-", "_")
    
    # Check if it's already a canonical category
    if normalized in REQUIREMENTS_MATRIX:
        return normalized
    
    # Check aliases
    if normalized in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[normalized]
    
    # Default to general
    return "general"


def get_requirements_for_category(category: str) -> Dict[str, Any]:
    """
    Get the requirements definition for a category.
    
    Args:
        category: Ticket category (will be normalized)
        
    Returns:
        Requirements dict for the category
    """
    canonical = get_canonical_category(category)
    return REQUIREMENTS_MATRIX.get(canonical, REQUIREMENTS_MATRIX["general"])


def get_all_categories() -> List[str]:
    """Get list of all canonical category names."""
    return list(REQUIREMENTS_MATRIX.keys())


# =============================================================================
# STRICT VALIDATION CATEGORIES
# =============================================================================

# These categories have strict field requirements that MUST be enforced.
# Categories not in this list will have constraint validation SKIPPED,
# allowing them to be processed as general/flexible tickets.
#
# Only add categories here that have:
# 1. Well-defined business rules
# 2. Clear required fields (e.g., warranty MUST have receipt)
# 3. Blocking requirements (can't proceed without certain info)
#
# Categories like "return_refund" are intentionally NOT included because:
# - They can be status inquiries (not actual return requests)
# - They can be credit questions (accounting issues)
# - Over-enforcement causes false positives

STRICT_VALIDATION_CATEGORIES: List[str] = [
    "warranty_claim",       # MUST have: receipt (proof of purchase)
    "missing_parts",        # MUST have: PO number to look up order
    "shipping_tracking",    # MUST have: PO number to track
    "replacement_parts",    # MUST have: model number to identify parts
]


def is_strictly_defined_category(category: str) -> bool:
    """
    Check if a category requires strict constraint validation.
    
    Categories in STRICT_VALIDATION_CATEGORIES have:
    - Well-defined required fields
    - Business rules that MUST be enforced
    
    Categories NOT in this list will be processed flexibly without
    strict field enforcement (like general tickets).
    
    Args:
        category: Ticket category (will be normalized)
        
    Returns:
        True if category requires strict validation, False otherwise
    """
    if not category:
        return False
    
    # Normalize the category
    normalized = category.lower().strip().replace(" ", "_").replace("-", "_")
    
    # Check if it's directly in strict list
    if normalized in STRICT_VALIDATION_CATEGORIES:
        return True
    
    # Check if its canonical form is in strict list
    canonical = get_canonical_category(normalized)
    if canonical in STRICT_VALIDATION_CATEGORIES:
        return True
    
    return False
