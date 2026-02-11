"""
Customer Lookup Node
Identifies customer type (DEALER vs END_CUSTOMER) based on email domain lookup.
Uses dealer_domain_service to check against the dealer domains CSV.
"""

import logging
import time
from typing import Dict, Any

from app.graph.state import TicketState
from app.config.constants import CustomerType
from app.services.dealer_domain_service import get_dealer_match_info

logger = logging.getLogger(__name__)
STEP_NAME = "6️⃣ CUSTOMER_LOOKUP"


def identify_customer_type(state: TicketState) -> Dict[str, Any]:
    """
    Determine customer type (DEALER / END_CUSTOMER) based on:
      - email domain lookup against dealer domains CSV

    Returns:
        Partial state update with:
            - customer_type
            - customer_metadata
            - audit_events
    """
    start_time = time.time()
    logger.info(f"{STEP_NAME} | ▶ Starting customer type identification")
    
    email = state.get("requester_email", "") or ""

    logger.info(f"{STEP_NAME} | 📥 Input: email='{email}'")

    customer_metadata: Dict[str, Any] = {"email": email}
    customer_type = CustomerType.END_CUSTOMER.value  # Default to end customer
    detection_reason = "default (no dealer domain match)"

    # ====================================================================
    # DEALER DOMAIN LOOKUP
    # Check if email domain is in the dealer domains list
    # ====================================================================
    if email:
        match_info = get_dealer_match_info(email)
        
        if match_info.get("is_dealer"):
            customer_type = CustomerType.DEALER.value
            customer_metadata["match_type"] = match_info.get("match_type")
            customer_metadata["matched_value"] = match_info.get("matched_value")
            detection_reason = f"Dealer domain match: {match_info.get('matched_value')} ({match_info.get('match_type')})"
            logger.info(f"{STEP_NAME} | 🏢 DEALER detected via {match_info.get('match_type')}: {match_info.get('matched_value')}")
        else:
            customer_metadata["email_domain"] = match_info.get("email_domain", "")
            logger.info(f"{STEP_NAME} | 👤 END_CUSTOMER (domain not in dealer list)")

    duration = time.time() - start_time
    logger.info(f"{STEP_NAME} | 🎯 Decision: customer_type='{customer_type}' (reason: {detection_reason})")
    logger.info(f"{STEP_NAME} | ✅ Complete in {duration:.2f}s")

    audit_events = state.get("audit_events", []) or []
    audit_events.append(
        {
            "event": "identify_customer_type",
            "customer_type": customer_type,
            "email": email,
            "detection_reason": detection_reason,
            "customer_metadata": customer_metadata,
        }
    )

    return {
        "customer_type": customer_type,
        "customer_metadata": customer_metadata,
        "audit_events": audit_events,
    }
