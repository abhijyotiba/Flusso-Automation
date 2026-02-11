"""
Customer Rules Node
Loads applicable business rules based on customer type (DEALER / END_CUSTOMER).
Implements rules from rules.txt for differentiated processing.
"""

import logging
import time
from typing import Dict, Any

from app.graph.state import TicketState
from app.utils.audit import add_audit_event
from app.config.constants import CustomerType

logger = logging.getLogger(__name__)
STEP_NAME = "7️⃣ CUSTOMER_RULES"


# ==============================================================================
# BUSINESS RULES FROM rules.txt
# ==============================================================================

DEALER_RULES = {
    # === IDENTITY ===
    "customer_type_label": "Dealer",
    "is_contracting_party": True,  # Dealer is the contracting party with Flusso
    
    # === REQUEST TYPES (Section 1) ===
    "request_types": [
        "tracking_order_status",
        "missing_items",
        "return",
        "exchange",
        "warranty",
        "restocking_fee_waiver",
        "spiff",
        "incoming_po",
        "account_statement_status"  # Credit hold, credit limit, declined card
    ],
    
    # === RETURNS & RESTOCKING (Section 6) ===
    "returns": {
        "allowed": True,
        "conditions": ["unused", "uninstalled", "original_packaging_preferred"],
        "opened_packaging": "may_be_accepted_at_discretion",
        "installed_items": "never_eligible",
        "freight_paid_by": "dealer",
        "rga_required": True,
        "credits_issued_after": "inspection",
        "additional_fees_for": ["damaged_packaging", "missing_components"],
        "restocking_fees": {
            "0-45_days": 0.15,    # 15%
            "45-90_days": 0.25,   # 25%
            "91-180_days": 0.50,  # 50%
            "over_180_days": None  # No returns accepted
        },
        "waiver_eligible": True,  # Rare, discretionary, typically 5-star dealers
        "waiver_note": "Must escalate for manual review. Never promise approval.",
        "waiver_criteria": "five_star_dealers_only"
    },
    
    # === MISSING ITEMS (Section 5) ===
    "missing_items": {
        "report_window_days": 45,
        "can_view_policy_docs": True,
        "can_view_photos": True,
        "can_reference_policy": True,
        "claims_after_window": "invalid",
        "ai_must_identify": ["invoice_number", "models_shipped", "quantities", "pre_shipment_photos"]
    },
    
    # === TRACKING & ORDER STATUS (Section 2) ===
    "tracking": {
        "self_service_url": "https://flussodealers.com/orderstatus/",
        "stock_check_url": "https://flussodealers.com/stockcheck",
        "provide_account_number": True,
        "can_see_full_order_details": True,
        "tracking_sent_to": ["po_submitter_email", "dealer_accounting_email"],
        "tracking_includes": ["packing_list", "tracking_numbers"],
        "ups_movement_delay": "may_not_appear_until_next_business_day",
        "instructions": "Provide dealer account number based on email domain. Instruct to use self-service portal with account number + order/PO number.",
        # IMPORTANT: PO details are available in internal system - NEVER ask dealer for dates/details
        "po_internal_lookup": True,
        "never_ask_for_when_po_present": ["purchase_date", "order_date", "po_date", "invoice_date", "ship_date", "delivery_date", "order_details"]
    },
    
    # === STOCK CHECK (Section 3) ===
    "stock": {
        "self_service_url": "https://flussodealers.com/stockcheck",
        "should_check_before_contacting": True,
        "if_item_not_listed": "may_email_flusso_for_inquiry"
    },
    
    # === FREIGHT & SHIPPING (Section 4) ===
    "freight": {
        "free_threshold_usd": 1000,
        "free_threshold_hawaii_usd": 2500,
        "free_threshold_canada_cad": 1700,
        "below_threshold": "freight_charges_invoiced",
        "freight_quote_response": "Please provide zip code and model numbers for a freight quote.",
        "handling_time_ground": "up_to_3_business_days_plus_2-3_freight",
        "handling_time_express": "up_to_2_business_days",
        "handling_time_guaranteed": False,
        "delays_may_occur_due_to": ["backorders", "credit_holds", "pending_customer_responses"],
        "express_same_day": False,
        "express_refund_on_delay": False,
        "drop_ship": "dealer_specifies_address_on_po"
    },
    
    # === DISCLOSURE RULES (Section 12) ===
    "disclosure": {
        "can_share_policy_docs": True,
        "can_share_restocking_details": True,
        "can_share_dealer_pricing": True,
        "can_mention_account_status": True,  # Credit hold, etc.
        "can_share_internal_approval_criteria": False,  # Never explain dealer ranking
        "can_share_warranty_process": True
    },
    
    # === WARRANTY (Section 8) ===
    "warranty": {
        "same_as_end_customer": True,
        "can_explain_full_process": True,
        "eligible_products": {
            "purchased_after": "2011-01-01",
            "from": ["flusso_direct", "authorized_dealer"],
            "installed_in": ["usa", "canada"]
        },
        "eligible_purchasers": "original_consumer_purchaser_only",
        "transferable": False,
        "periods": {
            "homeowner": "lifetime_while_original_owner_in_residence",
            "commercial": "2_years_from_purchase",
            "limited_components_1_year": [
                "drain_assemblies", "bathroom_hoses", "hand_helds",
                "kitchen_hand_sprays", "kitchen_hoses"
            ]
        },
        "intent": "repair_first_not_automatic_replacement",
        "covered_remedy": "replacement_of_defective_parts_or_finishes_free",
        "requirements": {
            "proof_of_purchase": True,
            "issue_description": True,
            "video_photos_required": True,
            "troubleshooting_required": True
        },
        "exclusions": [
            "installation_error",
            "product_misuse_or_abuse",
            "abrasive_alcohol_solvent_cleaners",
            "unauthorized_service"
        ],
        "third_party_parts": {
            "voids_warranty": False,  # Magnuson-Moss compliance
            "flusso_not_liable_for": ["improperly_installed", "incompatible", "unauthorized_components"],
            "recommendation": "use_genuine_flusso_parts"
        },
        "unauthorized_sellers": "warranty_does_not_apply"
    },
    
    # === COMPONENT-LEVEL WARRANTY (Section 9) ===
    "component_warranty": {
        "common_repairable_issues": [
            "leaking_hoses",
            "finish_defects_on_handles",
            "cartridge_issues",
            "aerator_issues",
            "loose_components"  # spout, hoses, handles
        ],
        "resolution": "replace_defective_component_no_charge",
        "full_replacement": "rare_determined_by_flusso_only"
    },
    
    # === PAYMENT TERMS (Section 10) ===
    "payment_terms": {
        "prepay_credit_card": {"fee": 0},
        "net_30_with_approval": {
            "check_or_ach": {"fee": 0},
            "credit_card": {"fee": 0.03}  # 3% fee
        }
    },
    
    # === MAP POLICY (Section 11) ===
    "map_policy": {
        "map_discount": 0.25,  # 25% off list price
        "enforcement": {
            "first_violation": "warning_email",
            "second_violation": "60_day_account_suspension",
            "third_violation": "account_closure"
        }
    },
    
    # === RESPONSE TONE ===
    "response_tone": {
        "formality": "professional",
        "can_reference_account": True,
        "can_reference_po_numbers": True,
        "salutation": "partner"
    }
}


END_CUSTOMER_RULES = {
    # === IDENTITY ===
    "customer_type_label": "End Customer",
    "is_contracting_party": False,  # Purchased from dealer, not direct from Flusso
    
    # === REQUEST TYPES (Section 1) ===
    "request_types": [
        "tracking_order_status",
        "missing_items",
        "warranty",
        "stock_availability"
        # Note: returns, restocking, spiff, account status NOT applicable
    ],
    
    # === RETURNS (Section 7) ===
    "returns": {
        "allowed": False,  # Cannot return directly to Flusso
        "redirect_to": "dealer",
        "message": "All return requests must be handled through your authorized dealer.",
        "can_mention_restocking_fees": False,  # NEVER
        "can_share_return_policy": False  # NEVER share dealer-only policies
    },
    
    # === MISSING ITEMS (Section 5) ===
    "missing_items": {
        "can_acknowledge_concern": True,
        "can_list_invoice_contents": True,
        "can_show_photos": True,
        "can_reference_45_day_window": True,
        "can_view_policy_docs": False,  # NEVER share dealer-only policy documents
        "ai_must_identify": ["invoice_number", "models_shipped", "quantities", "pre_shipment_photos"]
    },
    
    # === TRACKING & ORDER STATUS (Section 2) ===
    "tracking": {
        "can_share_tracking": True,
        "can_coordinate_with_dealer": True,
        "can_share_account_status": False,  # NEVER (credit hold, declined card, etc.)
        "can_guarantee_delivery": False,
        "can_override_carrier_delays": False,
        "unconfirmed_address_note": "May need to contact dealer for address verification.",
        "never_disclose_reasons": ["account_hold", "declined_credit_card", "credit_limit_reached"],
        "instructions": "Provide tracking if shipped. For order-specific availability, refer to dealer."
    },
    
    # === STOCK AVAILABILITY (Section 3) ===
    "stock": {
        "can_provide_general_availability": True,
        "reference_website": "https://flussofaucets.com",
        "website_shows": ["in_stock_status", "finish_availability", "stock_check"],
        "for_order_specific": "refer_to_dealer",
        "likely_hold_response": "Please contact your dealer for the latest status on your order.",
        "delay_reason_note": "If customer says order was placed days ago but no tracking, likely dealer account issue - refer to dealer without disclosing reason."
    },
    
    # === DISCLOSURE RULES (Section 12) ===
    "disclosure": {
        "can_share_policy_docs": False,  # NEVER
        "can_share_restocking_details": False,  # NEVER
        "can_share_dealer_pricing": False,
        "can_mention_account_status": False,  # NEVER (credit hold, etc.)
        "can_share_internal_approval_criteria": False,
        "never_share": [
            "dealer_only_policies",
            "restocking_fee_details",
            "internal_approval_criteria",
            "dealer_account_status"
        ]
    },
    
    # === WARRANTY (Section 8) ===
    "warranty": {
        "same_as_dealer": True,  # Warranty process is same for both
        "can_explain_full_process": True,
        "eligible_products": {
            "purchased_after": "2011-01-01",
            "from": ["flusso_direct", "authorized_dealer"],
            "installed_in": ["usa", "canada"]
        },
        "eligible_purchasers": "original_consumer_purchaser_only",
        "transferable": False,
        "periods": {
            "homeowner": "lifetime_while_original_owner_in_residence",
            "commercial": "2_years_from_purchase",
            "limited_components_1_year": [
                "drain_assemblies", "bathroom_hoses", "hand_helds",
                "kitchen_hand_sprays", "kitchen_hoses"
            ]
        },
        "intent": "repair_first_not_automatic_replacement",
        "requirements": {
            "proof_of_purchase": True,
            "issue_description": True,
            "video_photos_required": True,
            "troubleshooting_required": True
        },
        "installed_but_faulty": "not_eligible_for_return_must_use_warranty",
        "exclusions": [
            "installation_error",
            "product_misuse_or_abuse",
            "abrasive_alcohol_solvent_cleaners",
            "unauthorized_service"
        ],
        "unauthorized_sellers": "warranty_does_not_apply"
    },
    
    # === COMPONENT-LEVEL WARRANTY (Section 9) ===
    "component_warranty": {
        "common_repairable_issues": [
            "leaking_hoses",
            "finish_defects_on_handles",
            "cartridge_issues",
            "aerator_issues",
            "loose_components"
        ],
        "resolution": "replace_defective_component_no_charge",
        "full_replacement": "rare_determined_by_flusso_only"
    },
    
    # === RESPONSE TONE ===
    "response_tone": {
        "formality": "friendly_professional",
        "acknowledge_as": "valued_customer",
        "salutation": "customer"
    }
}





def load_customer_rules(state: TicketState) -> Dict[str, Any]:
    """
    Load business rules based on customer type (DEALER / END_CUSTOMER).

    Supported categories:
      - DEALER: Full policy access, can return, restocking fees apply
      - END_CUSTOMER: Limited disclosure, returns via dealer only
      
    Returns:
        - customer_rules: Dict with applicable business rules
    """
    start_time = time.time()
    logger.info(f"{STEP_NAME} | ▶ Starting customer rules lookup")

    customer_type = state.get("customer_type", CustomerType.END_CUSTOMER.value)
    customer_type = str(customer_type).upper().strip()

    logger.info(f"{STEP_NAME} | 📥 Input: customer_type='{customer_type}'")

    rules: Dict[str, Any] = {}
    rule_source = "none"

    # ----------------------- DEALER -----------------------
    if customer_type == CustomerType.DEALER.value:
        rules = DEALER_RULES.copy()
        rule_source = "dealer_rules"
        logger.info(f"{STEP_NAME} | 🏢 Dealer rules applied")

    # ----------------------- END CUSTOMER -----------------------
    elif customer_type == CustomerType.END_CUSTOMER.value:
        rules = END_CUSTOMER_RULES.copy()
        rule_source = "end_customer_rules"
        logger.info(f"{STEP_NAME} | 👤 End customer rules applied")

    # ----------------------- DEFAULT -----------------------
    if not rules:
        rules = END_CUSTOMER_RULES.copy()
        rule_source = "default_end_customer"
        logger.info(f"{STEP_NAME} | ⚠️ Unknown type, defaulting to end customer rules")

    duration = time.time() - start_time
    logger.info(f"{STEP_NAME} | ✅ Complete in {duration:.2f}s (source: {rule_source})")

    # ----------------------- AUDIT LOG -----------------------------
    audit_events = add_audit_event(
        state,
        "load_customer_rules",
        "RULES",
        {
            "customer_type": customer_type,
            "rule_source": rule_source,
            "rules_present": bool(rules),
            "rule_keys": list(rules.keys())
        },
    )["audit_events"]

    return {
        "customer_rules": rules,
        "audit_events": audit_events
    }


# ==============================================================================
# HELPER FUNCTIONS FOR RULE CHECKING
# ==============================================================================

def can_share_policy_docs(customer_type: str) -> bool:
    """Check if policy documents can be shared with this customer type."""
    return customer_type == CustomerType.DEALER.value


def can_process_return(customer_type: str) -> bool:
    """Check if returns can be processed directly."""
    return customer_type == CustomerType.DEALER.value


def get_return_message(customer_type: str) -> str:
    """Get appropriate return message based on customer type."""
    if customer_type == CustomerType.DEALER.value:
        return "Please review our return policy. Items must be unused and uninstalled. Restocking fees apply based on time since purchase."
    else:
        return "Returns must be handled through your authorized dealer. Please contact them for assistance."


def get_restocking_fee(days_since_purchase: int) -> float:
    """Get restocking fee percentage based on days since purchase."""
    if days_since_purchase <= 45:
        return 0.15
    elif days_since_purchase <= 90:
        return 0.25
    elif days_since_purchase <= 180:
        return 0.50
    else:
        return None  # No returns accepted


def get_tracking_response_guidance(customer_type: str) -> str:
    """Get guidance for tracking responses based on customer type."""
    if customer_type == CustomerType.DEALER.value:
        return """
For dealer tracking inquiries:
1. Provide the order status link: https://flussodealers.com/orderstatus/
2. Provide the dealer account number based on email domain
3. Instruct that dealer account number + order/PO number is required
"""
    else:
        return """
For end customer tracking inquiries:
- Provide tracking if the item has shipped
- For order-specific availability, refer to dealer
- NEVER mention account status (credit hold, declined card, etc.)
- If there appears to be a delay, suggest contacting the dealer
"""
