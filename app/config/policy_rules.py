"""
Policy Rules - Defines policy citations and validation rules.

This module defines STRUCTURED policy rules that are ENFORCED as constraints
on LLM responses. Unlike policy_service.py (which provides full policy text
for context), this module provides specific citations and validation rules
that MUST appear in responses.

Edit this file to add/modify policy rules.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# POLICY TYPES
# =============================================================================

class PolicyType(str, Enum):
    """Types of policies that can be applied."""
    WARRANTY = "warranty"
    RETURN = "return"
    MISSING_PARTS = "missing_parts"
    SHIPPING = "shipping"
    DEALER = "dealer"
    GENERAL = "general"


# =============================================================================
# POLICY RULES DEFINITIONS
# =============================================================================

POLICY_RULES: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # WARRANTY POLICIES
    # =========================================================================
    "warranty_standard": {
        "policy_id": "POL-001",
        "policy_type": PolicyType.WARRANTY.value,
        "name": "Standard Warranty",
        "citation": "Our standard warranty covers manufacturing defects for 1 year from the date of purchase.",
        "coverage_months": 12,
        "requires_date_check": True,
        "requires_receipt": True,
        "date_field": "purchase_date",
        "in_warranty_message": "Based on your purchase date, your product is within the 1-year warranty period. We'll be happy to assist with a replacement.",
        "out_warranty_message": "Based on your purchase date, your product is outside the 1-year warranty period. However, we can still help you with replacement parts at our standard pricing.",
        "unknown_date_message": "To verify your warranty coverage, please provide your proof of purchase showing the purchase date.",
        "applies_to": ["all"],
        "exceptions": ["cosmetic_damage", "misuse", "unauthorized_dealer"],
    },
    
    "hose_warranty": {
        "policy_id": "POL-002",
        "policy_type": PolicyType.WARRANTY.value,
        "name": "Extended Hose Warranty",
        "citation": "Hoses and supply lines are covered under our extended warranty for 2 years from the date of purchase.",
        "coverage_months": 24,
        "requires_date_check": True,
        "requires_receipt": True,
        "date_field": "purchase_date",
        "in_warranty_message": "Good news! Hoses have an extended 2-year warranty. Your product is within warranty coverage.",
        "out_warranty_message": "Hoses have a 2-year warranty period. Based on your purchase date, this is outside warranty coverage, but we can provide replacement hoses at our standard pricing.",
        "unknown_date_message": "Hoses have an extended 2-year warranty. Please provide your proof of purchase so we can verify coverage.",
        "applies_to": ["hose", "supply_line", "supply line", "braided", "water supply", "connector"],
        "product_keywords": ["hose", "supply line", "braided", "water supply", "connector hose"],
    },
    
    "lifetime_warranty": {
        "policy_id": "POL-003",
        "policy_type": PolicyType.WARRANTY.value,
        "name": "Lifetime Warranty",
        "citation": "This product includes our lifetime warranty against manufacturing defects for the original purchaser.",
        "coverage_months": None,  # Lifetime = no expiry
        "requires_date_check": False,
        "requires_receipt": True,
        "requires_original_purchaser": True,
        "in_warranty_message": "Your product is covered under our lifetime warranty for manufacturing defects.",
        "unknown_date_message": "Please provide your proof of purchase to verify original purchaser status for lifetime warranty coverage.",
        "applies_to": ["faucet_body", "valve_body"],
        "exceptions": ["finish_wear", "cosmetic_damage", "misuse"],
    },
    
    # =========================================================================
    # MISSING PARTS POLICY
    # =========================================================================
    "missing_parts_window": {
        "policy_id": "POL-004",
        "policy_type": PolicyType.MISSING_PARTS.value,
        "name": "Missing Parts Claim Window",
        "citation": "Missing parts must be reported within 45 days of delivery to qualify for free replacement.",
        "window_days": 45,
        "requires_date_check": True,
        "date_field": "delivery_date",
        "within_window_message": "Since you reported this within 45 days of delivery, we'll send the missing parts at no charge.",
        "outside_window_message": "Missing parts claims must be made within 45 days of delivery. Since this is outside that window, the parts can be purchased at our standard pricing.",
        "unknown_date_message": "Please provide your order number or delivery date so we can verify your eligibility for free replacement parts.",
        "applies_to": ["all"],
        "free_shipping": True,
    },
    
    # =========================================================================
    # RETURN POLICY
    # =========================================================================
    "return_policy": {
        "policy_id": "POL-005",
        "policy_type": PolicyType.RETURN.value,
        "name": "Return Policy",
        "citation": "Returns are accepted within 45 days of purchase for unused products in original packaging. A 15% restocking fee applies to opened items.",
        "window_days": 45,
        "restocking_fee_percent": 15,
        "requires_date_check": True,
        "requires_receipt": True,
        "date_field": "purchase_date",
        "conditions": ["unused", "original_packaging"],
        "within_window_message": "Your purchase is within our 45-day return window. Please note a 15% restocking fee applies to opened items.",
        "outside_window_message": "Returns are accepted within 45 days of purchase. Based on your purchase date, this is outside our standard return window.",
        "unknown_date_message": "Please provide your proof of purchase so we can verify your eligibility for return.",
        "rga_required": True,
        "rga_message": "To process your return, we'll need to issue an RGA (Return Goods Authorization) number. Once approved, you'll receive return shipping instructions.",
        "applies_to": ["all"],
        "non_returnable": ["custom_orders", "clearance_items", "final_sale"],
    },
    
    # =========================================================================
    # DEALER PROGRAM
    # =========================================================================
    "dealer_program": {
        "policy_id": "POL-006",
        "policy_type": PolicyType.DEALER.value,
        "name": "Dealer Program",
        "citation": "We welcome dealer and distributor partnerships. Our dealer program offers competitive pricing and dedicated support.",
        "requires_date_check": False,
        "requires_receipt": False,
        "application_message": "To become an authorized dealer, please provide your business information and we'll have our partnerships team reach out with program details.",
        "applies_to": ["dealer_inquiry"],
    },
}


# =============================================================================
# POLICY TRIGGERS - WHEN TO APPLY SPECIFIC POLICIES
# =============================================================================

POLICY_TRIGGERS: Dict[str, Dict[str, List[str]]] = {
    # Product-based triggers: if product text contains these keywords, apply these policies
    "product_keywords": {
        "hose": ["hose_warranty"],
        "supply line": ["hose_warranty"],
        "supply_line": ["hose_warranty"],
        "braided": ["hose_warranty"],
        "connector": ["hose_warranty"],
        "water supply": ["hose_warranty"],
    },
    
    # Category-based triggers (complements REQUIREMENTS_MATRIX)
    "category_triggers": {
        "warranty_claim": ["warranty_standard"],
        "product_issue": ["warranty_standard"],
        "missing_parts": ["missing_parts_window"],
        "return_refund": ["return_policy"],
        "replacement_parts": ["warranty_standard"],
        "dealer_inquiry": ["dealer_program"],
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_policy_rule(policy_key: str) -> Optional[Dict[str, Any]]:
    """
    Get a policy rule by its key.
    
    Args:
        policy_key: Policy key (e.g., "warranty_standard", "hose_warranty")
        
    Returns:
        Policy rule dict or None if not found
    """
    return POLICY_RULES.get(policy_key)


def get_policies_for_product(product_text: str) -> List[str]:
    """
    Get applicable policy keys based on product description/keywords.
    
    Args:
        product_text: Product description, name, or relevant text
        
    Returns:
        List of policy keys that apply
    """
    if not product_text:
        return []
    
    product_lower = product_text.lower()
    applicable = []
    
    for keyword, policies in POLICY_TRIGGERS["product_keywords"].items():
        if keyword in product_lower:
            for policy in policies:
                if policy not in applicable:
                    applicable.append(policy)
    
    return applicable


def get_policies_for_category(category: str) -> List[str]:
    """
    Get applicable policy keys based on ticket category.
    
    Args:
        category: Ticket category
        
    Returns:
        List of policy keys that apply
    """
    if not category:
        return []
    
    category_lower = category.lower().strip()
    return POLICY_TRIGGERS["category_triggers"].get(category_lower, [])


def get_policy_citation(policy_key: str) -> str:
    """
    Get the citation text for a policy.
    
    Args:
        policy_key: Policy key
        
    Returns:
        Citation text or empty string
    """
    policy = POLICY_RULES.get(policy_key, {})
    return policy.get("citation", "")


def get_all_policy_citations(policy_keys: List[str]) -> List[Dict[str, str]]:
    """
    Get all citations for a list of policies.
    
    Args:
        policy_keys: List of policy keys
        
    Returns:
        List of {policy_id, name, citation} dicts
    """
    citations = []
    for key in policy_keys:
        policy = POLICY_RULES.get(key)
        if policy:
            citations.append({
                "policy_id": policy.get("policy_id"),
                "policy_key": key,
                "name": policy.get("name"),
                "citation": policy.get("citation"),
            })
    return citations


def check_warranty_coverage(
    policy_key: str,
    purchase_date: Optional[str] = None,
    months_since_purchase: Optional[int] = None
) -> Dict[str, Any]:
    """
    Check warranty coverage status.
    
    Args:
        policy_key: Policy key (e.g., "warranty_standard")
        purchase_date: Purchase date string (for logging)
        months_since_purchase: Number of months since purchase
        
    Returns:
        {
            "is_covered": bool or None (if unknown),
            "message": str,
            "coverage_months": int,
        }
    """
    policy = POLICY_RULES.get(policy_key)
    if not policy:
        return {
            "is_covered": None,
            "message": "Unable to determine warranty coverage.",
            "coverage_months": None,
        }
    
    coverage_months = policy.get("coverage_months")
    
    # Lifetime warranty
    if coverage_months is None:
        return {
            "is_covered": True,
            "message": policy.get("in_warranty_message", "Product has lifetime warranty."),
            "coverage_months": None,
        }
    
    # Unknown purchase date
    if months_since_purchase is None:
        return {
            "is_covered": None,
            "message": policy.get("unknown_date_message", "Please provide purchase date."),
            "coverage_months": coverage_months,
        }
    
    # Check coverage
    is_covered = months_since_purchase <= coverage_months
    
    if is_covered:
        message = policy.get("in_warranty_message", f"Product is within {coverage_months}-month warranty.")
    else:
        message = policy.get("out_warranty_message", f"Product is outside {coverage_months}-month warranty.")
    
    return {
        "is_covered": is_covered,
        "message": message,
        "coverage_months": coverage_months,
        "months_since_purchase": months_since_purchase,
    }


def check_return_window(
    days_since_purchase: Optional[int] = None
) -> Dict[str, Any]:
    """
    Check return eligibility.
    
    Args:
        days_since_purchase: Days since purchase
        
    Returns:
        {
            "is_eligible": bool or None,
            "message": str,
            "restocking_fee": float,
        }
    """
    policy = POLICY_RULES.get("return_policy")
    if not policy:
        return {
            "is_eligible": None,
            "message": "Unable to determine return eligibility.",
            "restocking_fee": None,
        }
    
    window_days = policy.get("window_days", 45)
    restocking_fee = policy.get("restocking_fee_percent", 15)
    
    if days_since_purchase is None:
        return {
            "is_eligible": None,
            "message": policy.get("unknown_date_message"),
            "restocking_fee": restocking_fee,
            "window_days": window_days,
        }
    
    is_eligible = days_since_purchase <= window_days
    
    if is_eligible:
        message = policy.get("within_window_message")
    else:
        message = policy.get("outside_window_message")
    
    return {
        "is_eligible": is_eligible,
        "message": message,
        "restocking_fee": restocking_fee,
        "window_days": window_days,
        "days_since_purchase": days_since_purchase,
    }


def check_missing_parts_window(
    days_since_delivery: Optional[int] = None
) -> Dict[str, Any]:
    """
    Check missing parts claim eligibility.
    
    Args:
        days_since_delivery: Days since delivery
        
    Returns:
        {
            "is_eligible": bool or None,
            "message": str,
            "window_days": int,
        }
    """
    policy = POLICY_RULES.get("missing_parts_window")
    if not policy:
        return {
            "is_eligible": None,
            "message": "Unable to determine eligibility.",
            "window_days": None,
        }
    
    window_days = policy.get("window_days", 45)
    
    if days_since_delivery is None:
        return {
            "is_eligible": None,
            "message": policy.get("unknown_date_message"),
            "window_days": window_days,
        }
    
    is_eligible = days_since_delivery <= window_days
    
    if is_eligible:
        message = policy.get("within_window_message")
    else:
        message = policy.get("outside_window_message")
    
    return {
        "is_eligible": is_eligible,
        "message": message,
        "window_days": window_days,
        "days_since_delivery": days_since_delivery,
    }
