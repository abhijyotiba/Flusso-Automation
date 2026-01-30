"""
Ticket Intake Extractor Node
Deterministically extracts structured information from ticket before planning.

This node creates a `ticket_facts` record that:
1. Captures what information IS present (not what's required)
2. Is MUTABLE - react_agent and planner can update fields
3. Serves as HINTS to reduce redundant iterations
4. Parses product codes into model + finish components

Design Principles:
- Nothing here is "final" - downstream nodes can update anything
- Empty/None values are valid - means "not detected yet"
- Model and finish are tracked separately for accuracy
- This is a HINT system, not a constraint system
"""

import re
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

from app.graph.state import TicketState
from app.utils.audit import add_audit_event

logger = logging.getLogger(__name__)
STEP_NAME = "🔍 TICKET_EXTRACTOR"


# =============================================================================
# FINISH CODE DICTIONARY
# =============================================================================
FINISH_CODES = {
    "CP": "Chrome",
    "BN": "Brushed Nickel PVD",
    "PN": "Polished Nickel PVD",
    "MB": "Matte Black",
    "SB": "Satin Brass PVD",
    "BB": "Brushed Bronze PVD",
    "GM": "Gunmetal",
    "WH": "White",
    "GD": "Gold",
    "RG": "Rose Gold",
    "AB": "Antique Brass",
    "PB": "Polished Brass",
    "ORB": "Oil Rubbed Bronze",  # 3-letter code
    "SS": "Stainless Steel",
    "MG": "Matte Grey",
    "CG": "Champagne Gold",
}

# 2-letter finish codes for suffix detection
TWO_LETTER_FINISHES = {k: v for k, v in FINISH_CODES.items() if len(k) == 2}


# =============================================================================
# PRODUCT CODE PATTERNS
# =============================================================================
PRODUCT_PATTERNS = [
    # TRM.TVH.0211BB, 10.FGC.4003CP format (letters.letters.digits + optional finish)
    r'\b(\d{1,3}\.?[A-Z]{2,4}\.[A-Z]{0,4}\.?\d{3,4}[A-Z]{0,3})\b',
    
    # 100.1170CP, 160.2420MB format (digits.digits + optional finish)
    r'\b(\d{3}\.\d{3,4}[A-Z]{0,3})\b',
    
    # PBV1005A, PBV2105, PBV.2105 format
    r'\b(PBV\.?\d{4}[A-Z]?[A-Z]{0,2})\b',
    
    # HS6270MB, DKM2420BN format (letters + digits + optional finish)
    r'\b([A-Z]{2,4}\d{4}[A-Z]{0,3})\b',
    
    # TRM.TVH.0211 format (letters.letters.digits)
    r'\b([A-Z]{2,4}\.[A-Z]{2,4}\.\d{3,4}[A-Z]{0,2})\b',
    
    # UF.2102, CFB.2250 format
    r'\b([A-Z]{2,3}\.\d{4}[A-Z]{0,2})\b',
    
    # K.1230-2229 format (with hyphen)
    r'\b([A-Z]\.\d{4}-\d{4}[A-Z]{0,2})\b',
    
    # General format: 2-4 letters followed by period and numbers
    r'\b([A-Z]{2,4}\.\d{3,5}[A-Z]{0,3})\b',
]

# Part number patterns (often include hyphens)
PART_NUMBER_PATTERNS = [
    r'\b(\d{7}-\d{3}[A-Z]?)\b',        # 8002048-122
    r'\b(\d{7}-\d{3})\b',               # 6032029-383
    r'\b([A-Z]{2,4}\d{4}[A-Z]?-\d{4})\b',  # PBV1005A-1991
    r'\b(\d{4}-\d{4})\b',               # 9853-1234
]

# Address indicators
ADDRESS_KEYWORDS = [
    r'\baddress\b', r'\bship\s*to\b', r'\bsend\s*to\b', r'\bdeliver\s*to\b',
    r'\bstreet\b', r'\bavenue\b', r'\bave\b', r'\bblvd\b', r'\bdrive\b',
    r'\bcity\b', r'\bstate\b', r'\bzip\b', r'\bpostal\b', r'\bzipcode\b',
    r'\b\d{5}(-\d{4})?\b',  # ZIP code pattern
]

# Receipt/proof of purchase indicators
RECEIPT_KEYWORDS = [
    r'\breceipt\b', r'\binvoice\b', r'\bproof\s*of\s*purchase\b',
    r'\border\s*confirmation\b', r'\bpurchase\s*date\b', r'\bbought\b',
    r'\bpurchased\b', r'\border\s*number\b', r'\btransaction\b',
]

# PO/Order number indicators  
PO_KEYWORDS = [
    r'\bpo\s*#', r'\bpo\s*:', r'\bpo\s*\d', r'\bpurchase\s*order\b',
    r'\border\s*#', r'\border\s*number\b', r'\border\s*:\s*\d',
    r'\bp\.o\.\b', r'\bpo\b\s*\d{4,}',
]

# Finish name mentions (text, not codes)
FINISH_NAMES = [
    r'\bchrome\b', r'\bbrushed\s*nickel\b', r'\bpolished\s*nickel\b',
    r'\bmatte\s*black\b', r'\bsatin\s*brass\b', r'\bbrushed\s*bronze\b',
    r'\bgunmetal\b', r'\boil\s*rubbed\s*bronze\b', r'\bstainless\b',
    r'\bbrass\b', r'\bnickel\b', r'\bgold\b',
]


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================
def parse_product_code(full_code: str) -> Dict[str, Any]:
    """
    Parse a product code into model number and finish components.
    
    Examples:
        "TRM.TVH.0211BB" → {"model": "TRM.TVH.0211", "finish_code": "BB", "finish_name": "Brushed Bronze PVD"}
        "10.FGC.4003CP" → {"model": "10.FGC.4003", "finish_code": "CP", "finish_name": "Chrome"}
        "PBV1005A" → {"model": "PBV1005A", "finish_code": None, "finish_name": None}
    
    Args:
        full_code: The full product code/SKU
        
    Returns:
        Dict with full_sku, model, finish_code, finish_name
    """
    full_code = full_code.strip().upper()
    
    # Handle edge cases
    if not full_code or len(full_code) < 3:
        return {
            "full_sku": full_code,
            "model": full_code,
            "finish_code": None,
            "finish_name": None
        }
    
    # Check if last 2 characters are a known finish code
    potential_finish_2 = full_code[-2:]
    if potential_finish_2 in TWO_LETTER_FINISHES:
        # Verify the character before the finish is not a letter (to avoid false positives)
        # e.g., "PROBLEM" shouldn't match "EM" as finish
        if len(full_code) > 2:
            char_before = full_code[-3]
            # If char before is a digit or period, it's likely a real finish code
            if char_before.isdigit() or char_before == '.' or char_before == 'A':
                return {
                    "full_sku": full_code,
                    "model": full_code[:-2],
                    "finish_code": potential_finish_2,
                    "finish_name": TWO_LETTER_FINISHES[potential_finish_2]
                }
    
    # Check for 3-letter finish codes (less common)
    if len(full_code) >= 4:
        potential_finish_3 = full_code[-3:]
        if potential_finish_3 in FINISH_CODES:
            return {
                "full_sku": full_code,
                "model": full_code[:-3],
                "finish_code": potential_finish_3,
                "finish_name": FINISH_CODES[potential_finish_3]
            }
    
    # No finish code detected
    return {
        "full_sku": full_code,
        "model": full_code,
        "finish_code": None,
        "finish_name": None
    }


def extract_product_codes(text: str) -> List[Dict[str, Any]]:
    """
    Extract product codes from text and parse into model + finish components.
    
    Args:
        text: The text to search
        
    Returns:
        List of parsed product codes, each with full_sku, model, finish_code, finish_name
    """
    if not text:
        return []
    
    found_codes = set()
    text_upper = text.upper()
    
    for pattern in PRODUCT_PATTERNS:
        try:
            matches = re.findall(pattern, text_upper)
            found_codes.update(matches)
        except re.error as e:
            logger.warning(f"{STEP_NAME} | Regex error for pattern {pattern}: {e}")
    
    # Parse each found code
    results = []
    seen_models = set()
    
    for code in found_codes:
        parsed = parse_product_code(code)
        # Deduplicate by model (not full SKU) to avoid duplicates like "100.1170" and "100.1170CP"
        model = parsed["model"]
        if model not in seen_models:
            results.append(parsed)
            seen_models.add(model)
    
    return results


def extract_part_numbers(text: str) -> List[str]:
    """Extract potential part numbers from text."""
    if not text:
        return []
    
    found_parts = set()
    text_upper = text.upper()
    
    for pattern in PART_NUMBER_PATTERNS:
        try:
            matches = re.findall(pattern, text_upper)
            found_parts.update(matches)
        except re.error:
            pass
    
    return list(found_parts)


def detect_keyword_presence(text: str, patterns: List[str]) -> bool:
    """Check if any of the patterns match in the text."""
    if not text:
        return False
    
    text_lower = text.lower()
    
    for pattern in patterns:
        try:
            if re.search(pattern, text_lower):
                return True
        except re.error:
            pass
    
    return False


def extract_address(text: str) -> Tuple[Optional[str], float]:
    """
    Attempt to extract an address from text.
    Returns (extracted_address, confidence)
    
    This is a basic extraction - confidence is low because
    address parsing is complex and should be verified.
    """
    if not text:
        return None, 0.0
    
    # Look for common address patterns
    # US address: number street, city, state ZIP
    address_pattern = r'(\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Drive|Dr|Road|Rd|Lane|Ln|Way|Court|Ct)[,\s]+[A-Za-z\s]+[,\s]+[A-Z]{2}\s+\d{5}(?:-\d{4})?)'
    
    match = re.search(address_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip(), 0.6  # Medium confidence
    
    # Simpler pattern: just city, state ZIP
    simple_pattern = r'([A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)'
    match = re.search(simple_pattern, text)
    if match:
        return match.group(1).strip(), 0.4  # Lower confidence
    
    return None, 0.0


def extract_finish_mentions(text: str) -> List[str]:
    """Extract mentioned finish names from text."""
    if not text:
        return []
    
    found_finishes = []
    text_lower = text.lower()
    
    finish_map = {
        'chrome': 'Chrome',
        'brushed nickel': 'Brushed Nickel PVD',
        'polished nickel': 'Polished Nickel PVD',
        'matte black': 'Matte Black',
        'satin brass': 'Satin Brass PVD',
        'brushed bronze': 'Brushed Bronze PVD',
        'gunmetal': 'Gunmetal',
        'oil rubbed bronze': 'Oil Rubbed Bronze',
        'stainless': 'Stainless Steel',
        'gold': 'Gold',
    }
    
    for keyword, full_name in finish_map.items():
        if keyword in text_lower:
            found_finishes.append(full_name)
    
    return found_finishes


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================
def extract_ticket_facts(state: TicketState) -> Dict[str, Any]:
    """
    Extract structured facts from ticket before planning.
    
    This creates a `ticket_facts` record that captures what information
    IS present in the ticket. All fields are HINTS that can be updated
    by planner or react_agent.
    
    Args:
        state: Current ticket state (with ticket_text, ticket_subject, etc.)
        
    Returns:
        Dict containing ticket_facts and audit_events
    """
    start_time = time.time()
    
    logger.info(f"{'='*60}")
    logger.info(f"{STEP_NAME} | Starting ticket facts extraction")
    logger.info(f"{'='*60}")
    
    # Get ticket content
    ticket_text = state.get("ticket_text", "") or ""
    ticket_subject = state.get("ticket_subject", "") or ""
    combined_text = f"{ticket_subject}\n{ticket_text}"
    
    # Get metadata from state (already extracted by fetch_ticket)
    customer_name = state.get("requester_name", "") or ""
    requester_email = state.get("requester_email", "") or ""
    
    # Get attachment info
    ticket_images = state.get("ticket_images", []) or []
    ticket_attachments = state.get("ticket_attachments", []) or []
    
    # Check for video attachments
    has_video = any(
        att.get("content_type", "").startswith("video")
        for att in ticket_attachments
    )
    
    # Check for document attachments
    has_document_attachments = any(
        not att.get("content_type", "").startswith(("image", "video"))
        for att in ticket_attachments
    )
    
    # =================================================================
    # TIER 1: DETERMINISTIC EXTRACTION
    # =================================================================
    
    # Boolean presence flags
    has_address = detect_keyword_presence(combined_text, ADDRESS_KEYWORDS)
    has_receipt = detect_keyword_presence(combined_text, RECEIPT_KEYWORDS)
    has_po = detect_keyword_presence(combined_text, PO_KEYWORDS)
    has_photos = len(ticket_images) > 0
    
    # Extract address (with confidence)
    extracted_address, address_confidence = extract_address(combined_text)
    
    # Extract product codes (parsed into model + finish)
    raw_product_codes = extract_product_codes(combined_text)
    
    # Extract part numbers
    raw_part_numbers = extract_part_numbers(combined_text)
    
    # Extract finish mentions (text-based)
    raw_finish_mentions = extract_finish_mentions(combined_text)
    
    # =================================================================
    # BUILD TICKET_FACTS RECORD
    # =================================================================
    
    # Determine has_model_number based on product codes found
    has_model_number = len(raw_product_codes) > 0
    
    ticket_facts = {
        # ----- TIER 1: DETERMINISTIC (set by this extractor) -----
        # Boolean presence flags
        "has_model_number": has_model_number,  # True if any product codes were found
        "has_address": has_address,
        "has_receipt": has_receipt,
        "has_po": has_po,
        "has_video": has_video,
        "has_photos": has_photos,
        "has_document_attachments": has_document_attachments,
        
        # Metadata from Freshdesk (reliable)
        "customer_name": customer_name,
        "requester_email": requester_email,
        
        # Address extraction (editable)
        "extracted_address": extracted_address,
        "address_confidence": address_confidence,
        "address_needs_confirmation": has_address and address_confidence < 0.7,
        
        # Raw extraction candidates (NOT finalized)
        "raw_product_codes": raw_product_codes,  # List of {full_sku, model, finish_code, finish_name}
        "raw_part_numbers": raw_part_numbers,
        "raw_finish_mentions": raw_finish_mentions,
        
        # ----- TIER 2: LLM-VERIFIED (set by planner) -----
        "planner_verified": False,
        "planner_verified_models": [],
        "planner_verified_finishes": [],
        "planner_corrections": {},
        
        # ----- TIER 3: TOOL-CONFIRMED (set by react_agent) -----
        "confirmed_model": None,
        "confirmed_model_source": None,  # "catalog", "ocr", "vision", "text"
        "confirmed_model_confidence": 0.0,
        "confirmed_finish": None,
        "confirmed_finish_name": None,
        "confirmed_parts": [],
        
        # ----- METADATA -----
        "extraction_version": "1.0",
        "extracted_at": time.time(),
        "last_updated_at": time.time(),
        "last_updated_by": "ticket_extractor",
        "update_history": []
    }
    
    # =================================================================
    # LOGGING
    # =================================================================
    duration = time.time() - start_time
    
    logger.info(f"{STEP_NAME} | ✅ Extraction completed in {duration:.2f}s")
    logger.info(f"{STEP_NAME} | Presence flags: address={has_address}, receipt={has_receipt}, po={has_po}, photos={has_photos}, video={has_video}")
    
    if raw_product_codes:
        models = [p["model"] for p in raw_product_codes]
        finishes = [p["finish_code"] for p in raw_product_codes if p.get("finish_code")]
        logger.info(f"{STEP_NAME} | 📦 Product codes found: {models}")
        if finishes:
            logger.info(f"{STEP_NAME} | 🎨 Finish codes detected: {finishes}")
    
    if raw_part_numbers:
        logger.info(f"{STEP_NAME} | 🔧 Part numbers found: {raw_part_numbers}")
    
    if raw_finish_mentions:
        logger.info(f"{STEP_NAME} | 🎨 Finish mentions: {raw_finish_mentions}")
    
    if extracted_address:
        logger.info(f"{STEP_NAME} | 📍 Address extracted (confidence={address_confidence:.0%}): {extracted_address[:50]}...")
    
    # =================================================================
    # RETURN STATE UPDATES
    # =================================================================
    return {
        "ticket_facts": ticket_facts,
        "audit_events": add_audit_event(
            state,
            event="ticket_extractor",
            event_type="SUCCESS",
            details={
                "has_address": has_address,
                "has_receipt": has_receipt,
                "has_po": has_po,
                "has_photos": has_photos,
                "has_video": has_video,
                "product_codes_count": len(raw_product_codes),
                "part_numbers_count": len(raw_part_numbers),
                "finish_mentions_count": len(raw_finish_mentions),
                "duration_seconds": duration,
            }
        )["audit_events"]
    }


# =============================================================================
# HELPER FUNCTIONS FOR OTHER NODES
# =============================================================================
def update_ticket_facts(
    current_facts: Dict[str, Any],
    updates: Dict[str, Any],
    updated_by: str
) -> Dict[str, Any]:
    """
    Update ticket_facts with new information while preserving audit trail.
    
    This function should be used by planner and react_agent to update facts.
    
    Args:
        current_facts: Current ticket_facts dict
        updates: Dict of fields to update
        updated_by: Who is making the update ("planner", "react_agent", etc.)
        
    Returns:
        Updated ticket_facts dict
    """
    if not current_facts:
        current_facts = {}
    
    # Create copy to avoid mutation
    updated_facts = current_facts.copy()
    
    # Track what changed
    changes = {}
    for key, new_value in updates.items():
        old_value = updated_facts.get(key)
        if old_value != new_value:
            changes[key] = {"old": old_value, "new": new_value}
            updated_facts[key] = new_value
    
    # Update metadata
    updated_facts["last_updated_at"] = time.time()
    updated_facts["last_updated_by"] = updated_by
    
    # Add to history
    if changes:
        history = updated_facts.get("update_history", [])
        history.append({
            "timestamp": time.time(),
            "updated_by": updated_by,
            "changes": changes
        })
        updated_facts["update_history"] = history[-10:]  # Keep last 10 updates
    
    return updated_facts


def get_model_candidates_from_facts(ticket_facts: Dict[str, Any]) -> List[str]:
    """
    Get best model candidates from ticket_facts in priority order.
    
    Priority:
    1. confirmed_model (if set)
    2. planner_verified_models (if any)
    3. raw_product_codes models
    
    Returns:
        List of model numbers to try
    """
    if not ticket_facts:
        return []
    
    models = []
    
    # Priority 1: Confirmed model
    if ticket_facts.get("confirmed_model"):
        models.append(ticket_facts["confirmed_model"])
    
    # Priority 2: Planner-verified models
    planner_models = ticket_facts.get("planner_verified_models", [])
    for m in planner_models:
        if m not in models:
            models.append(m)
    
    # Priority 3: Raw extracted models
    raw_codes = ticket_facts.get("raw_product_codes", [])
    for code in raw_codes:
        model = code.get("model")
        if model and model not in models:
            models.append(model)
    
    return models
