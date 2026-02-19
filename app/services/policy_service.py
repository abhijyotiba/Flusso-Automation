"""
Policy & Process Guide Service
Fetches and caches company policies from Google Docs for planning decisions.
Similar pattern to product_catalog_cache.py
"""

import requests
import time
import threading
import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# ===============================
# CONFIG
# ===============================
# Google Docs export URL (replace YOUR_DOC_ID with actual document ID)
# Format: https://docs.google.com/document/d/YOUR_DOC_ID/export?format=txt
# You can also use: /export?format=html for HTML format
GOOGLE_DOCS_POLICY_URL = "https://docs.google.com/document/d/1NYWE1ZnSQDgdRW0XtUt4eMggvJp7Rcp9XRrKXMQvfF0/export?format=txt"

# Refresh every 6 hours (policies don't change often)
REFRESH_INTERVAL_SECONDS = 21600  # 6 hours

# ===============================
# GLOBAL STATE
# ===============================
POLICY_CACHE: Dict[str, str] = {}  # category -> policy text
FULL_POLICY_TEXT: str = ""
POLICY_SECTIONS: Dict[str, Dict[str, Any]] = {}  # Parsed sections
LAST_REFRESH = 0
IS_REFRESHING = False
LOCK = threading.Lock()

# ===============================
# POLICY CATEGORIES (for quick matching)
# ===============================
POLICY_CATEGORIES = {
    "warranty": ["warranty", "warranty_claim", "defect", "broken", "malfunction", "not working"],
    "return": ["return", "refund", "return_refund", "rga", "send back", "money back"],
    "replacement_parts": ["replacement", "replacement_parts", "spare part", "part needed"],
    "missing_parts": ["missing", "missing_parts", "not included", "incomplete", "parts missing"],
    "shipping": ["shipping", "shipping_tracking", "delivery", "tracking", "where is my order"],
    "installation": ["installation", "installation_help", "install", "setup", "mounting", "how to"],
    "pricing": ["pricing", "pricing_request", "msrp", "price", "cost", "quote"],
    "dealer": ["dealer", "dealer_inquiry", "partnership", "account", "resale", "distributor"],
    "finish_color": ["finish", "finish_color", "color", "chrome", "brass", "nickel", "matte"],
    "feedback": ["feedback", "feedback_suggestion", "suggestion", "idea", "improvement"],
    "general": ["general", "product_inquiry", "question", "information"]
}

# ===============================
# CATEGORY TIPS - LLM Behavioral Guidance
# ===============================
# These are quick do/don't rules for the LLM when generating responses.
# Aligned with official policy rules - policy_service is the SINGLE SOURCE OF TRUTH.
# 
# Structure:
#   - do: Things the LLM MUST include/do
#   - dont: Things the LLM must NEVER do
#   - required_fields: What info is required before processing (for constraint validation)
#   - sample_response: Example response format (optional)
#
# IMPORTANT: Keep these aligned with LOCAL_FALLBACK_POLICY and Google Docs policy.
# ===============================

CATEGORY_TIPS = {
    # ===== FULL SUPPORT CATEGORIES (need product identification) =====
    
    "product_issue": {
        "label": "PRODUCT ISSUE / DEFECT",
        "do": [
            "Identify the product if possible from model number or description",
            "Reference warranty policy based on customer type (residential=lifetime, commercial=2yr)",
            "Request VIDEO evidence (preferred over photos) showing the defect",
            "Verify PO/invoice number or proof of purchase",
            "Confirm shipping address for replacement parts",
            "Ensure troubleshooting steps are completed before approval",
        ],
        "dont": [
            "Approve replacement without proof of purchase",
            "Ship replacements without confirmed shipping address",
            "Process warranty without video/photo evidence",
            "Promise full product replacement (parts replacement is standard)",
            "Invent warranty durations - only lifetime/2yr/1yr(limited components) exist",
        ],
        "required_fields": ["proof_of_purchase", "video_or_photo", "shipping_address"],
        "sample_response": None,
    },
    
    "replacement_parts": {
        "label": "REPLACEMENT PARTS REQUEST",
        "do": [
            "Identify the specific part needed (cartridge, handle, hose, etc.)",
            "Verify if under warranty (residential=lifetime, commercial=2yr, limited components=1yr)",
            "Request VIDEO showing the issue (preferred over photos)",
            "Verify PO/invoice number or proof of purchase",
            "Confirm shipping address for replacement delivery",
            "Check if it's a limited-warranty component (drains, hoses, hand-helds, hand sprays = 1yr only)",
        ],
        "dont": [
            "Approve without proof of purchase",
            "Ship without confirmed shipping address",
            "Process without video/photo evidence of defect",
            "Assume 1-year warranty for cartridges (cartridges get full warranty: lifetime/2yr)",
        ],
        "required_fields": ["proof_of_purchase", "video_or_photo", "shipping_address"],
        "sample_response": None,
    },
    
    "warranty_claim": {
        "label": "WARRANTY CLAIM",
        "do": [
            "Verify warranty eligibility (purchased after Jan 2011, from authorized dealer)",
            "Determine warranty period: residential=lifetime, commercial=2yr, limited components=1yr",
            "Request VIDEO evidence (preferred) showing the defect",
            "Verify PO/invoice number (invoice number is sufficient)",
            "Ensure troubleshooting steps are completed BEFORE approval",
            "Clarify that warranty covers PARTS replacement, not full product replacement",
        ],
        "dont": [
            "Approve without proof of purchase (invoice number is sufficient)",
            "Process without video/photo evidence",
            "Promise full product replacement (rare, internal decision only)",
            "Invent warranty durations like 'standard 1-year' or '2-year extended'",
            "Ask for purchase DATE if PO/invoice number is provided (we can look it up)",
            "Assign 2-year coverage to hoses (hoses are 1-year limited components)",
        ],
        "required_fields": ["proof_of_purchase", "video_or_photo", "shipping_address"],
        "sample_response": None,
    },
    
    "missing_parts": {
        "label": "MISSING PARTS FROM ORDER",
        "do": [
            "Verify order number/PO number in system",
            "Confirm exactly which parts are missing (reference packing list)",
            "Ship missing parts immediately - no questions asked within 45 days",
            "Acknowledge the inconvenience",
        ],
        "dont": [
            "Ask for receipt/proof of purchase (we have the order record)",
            "Ask for photos of missing parts (we know what was supposed to ship)",
            "Delay - this should be fast resolution",
            "Require customer to return anything",
        ],
        "required_fields": ["order_number"],
        "sample_response": "We apologize for the inconvenience. We've verified your order and will ship the missing [parts] right away. You should receive them within [X] business days.",
    },
    
    # ===== PRODUCT INFORMATION CATEGORIES =====
    
    "product_inquiry": {
        "label": "PRODUCT INQUIRY / STOCK AVAILABILITY",
        "do": [
            "Provide product specifications if found",
            "Include product page URL for real-time inventory: https://www.flussofaucets.com/products/[MODEL]-[name]/",
            "State lead time: '5-7 business days from the day we receive the purchase order'",
            "Translate 'lead time of 0' to 'in stock'",
            "Mention shipping may vary for California or New York",
        ],
        "dont": [
            "Ask for photos, videos, or receipts (not needed for stock inquiries)",
            "Say 'lead time of 0 days' to customer (say 'in stock' instead)",
            "Guess availability - direct to product page for real-time info",
        ],
        "required_fields": [],
        "sample_response": "The [model] is in stock - our site will show current updated inventory: [product_url]\nLead time is about 5-7 business days from the day we receive the purchase order.",
    },
    
    "installation_help": {
        "label": "INSTALLATION HELP",
        "do": [
            "Provide installation guidance from product documentation",
            "Reference installation manuals/videos if available",
            "Identify the specific product if model number provided",
            "Offer to send installation instructions via email",
        ],
        "dont": [
            "Ask for receipts or proof of purchase (not relevant for installation help)",
            "Require photos unless needed to identify the product",
            "Provide incorrect installation advice - defer to official docs",
        ],
        "required_fields": [],
        "sample_response": None,
    },
    
    "finish_color": {
        "label": "FINISH / COLOR INQUIRY",
        "do": [
            "Provide available finish options for the product",
            "List finish codes (e.g., CP=Chrome, BN=Brushed Nickel, MB=Matte Black)",
            "Direct to product page for visual finish samples",
            "Clarify if certain finishes are special order or have longer lead times",
        ],
        "dont": [
            "Ask for receipts or proof of purchase",
            "Guess finish availability - check product catalog",
            "Confuse finish codes between product lines",
        ],
        "required_fields": [],
        "sample_response": None,
    },
    
    # ===== INFORMATION REQUEST CATEGORIES =====
    
    "pricing_request": {
        "label": "PRICING / MSRP REQUEST",
        "do": [
            "Provide MSRP/pricing if found in search results",
            "Reference the specific part numbers customer asked about",
            "If pricing not found, say we will follow up with pricing information",
        ],
        "dont": [
            "Ask for product photos or receipts (not needed for pricing)",
            "Ask for model numbers (customer already provided part numbers)",
            "Share dealer-specific pricing with end customers",
        ],
        "required_fields": [],
        "sample_response": None,
    },
    
    "dealer_inquiry": {
        "label": "DEALER / PARTNERSHIP INQUIRY",
        "do": [
            "Acknowledge their interest in becoming a Flusso dealer/partner",
            "If they submitted documents (application, resale certificate), acknowledge receipt",
            "Provide next steps for application review",
            "Mention typical approval timeline if known",
        ],
        "dont": [
            "Ask for product photos or model numbers (irrelevant)",
            "Share confidential dealer pricing or terms before approval",
            "Promise approval - applications are reviewed internally",
        ],
        "required_fields": [],
        "sample_response": "Thank you for your interest in becoming a Flusso dealer! We've received your [application/documents] and our team will review it. You can expect to hear back within [X] business days.",
    },
    
    # ===== SPECIAL HANDLING CATEGORIES =====
    
    "shipping_tracking": {
        "label": "SHIPPING / TRACKING",
        "do": [
            "Provide tracking information if available",
            "Reference order number/PO for lookup",
            "For dealers: direct to self-service portal https://flussodealers.com/orderstatus/",
            "Acknowledge their inquiry and provide status update",
        ],
        "dont": [
            "Ask for product photos (not relevant for shipping)",
            "Mention account status, credit holds, or payment issues to end customers",
            "Share internal order processing details",
        ],
        "required_fields": ["order_number"],
        "sample_response": None,
    },
    
    "return_refund": {
        "label": "RETURN / REFUND REQUEST",
        "do": [
            "Collect PO/order number for verification",
            "Ask for reason for return (defective? wrong item? no longer needed?)",
            "Request photo of product condition if defective",
            "Reference return policy timeframes (0-45d=15%, 46-90d=25%, 91-180d=50%, 180+=no returns)",
            "State that RGA number will be issued after verification",
            "Mention original packaging required",
        ],
        "dont": [
            "Approve return without required information",
            "Accept returns on installed products (must go through warranty process)",
            "Accept returns on custom/special orders or clearance items",
            "Process returns beyond 180 days",
        ],
        "required_fields": ["order_number", "reason_for_return", "photo_if_defective"],
        "sample_response": None,
    },
    
    "feedback_suggestion": {
        "label": "FEEDBACK / SUGGESTION",
        "do": [
            "Thank the customer for their feedback",
            "Acknowledge their suggestion professionally",
            "State that feedback is shared with the product team",
        ],
        "dont": [
            "Ask for product photos or receipts (not relevant)",
            "Promise that suggestions will be implemented",
            "Dismiss or argue with feedback",
        ],
        "required_fields": [],
        "sample_response": "Thank you for sharing your feedback! We appreciate you taking the time to let us know. We'll share this with our product team for consideration.",
    },
    
    "general": {
        "label": "GENERAL INQUIRY / ACCOUNT UPDATE",
        "do": [
            "Acknowledge the customer's request professionally",
            "If account/address update: confirm receipt and note that records will be updated",
            "If general question: answer directly or acknowledge you'll look into it",
            "Keep response brief and professional",
        ],
        "dont": [
            "Ask for product model numbers or photos (not product-related)",
            "Assume this is a product support request",
            "Overcomplicate simple administrative requests",
        ],
        "required_fields": [],
        "sample_response": None,
    },
}

# ===============================
# LOCAL FALLBACK POLICY
# ===============================
LOCAL_FALLBACK_POLICY = """
# FLUSSO FAUCETS - OFFICIAL WARRANTY POLICY & PROCESSING LOGIC

## 1️⃣ WARRANTY SCOPE & ELIGIBILITY

### Eligible Products
Warranty applies ONLY to:
- Flusso products purchased after January 1, 2011
- Purchased directly from Flusso or an authorized Flusso dealer
- Installed in the United States or Canada

### Eligible Purchaser
- Original consumer purchaser ONLY
- Warranty is NON-TRANSFERABLE

---

## 2️⃣ WARRANTY PERIODS (AUTHORITATIVE STRUCTURE)

⚠️ THERE ARE ONLY THREE VALID WARRANTY DURATIONS IN THE SYSTEM:

### A. Homeowners (Residential Use) → LIFETIME
Valid for as long as:
- The original purchaser owns the product
- The product remains installed in the original residence
This functions as a LIFETIME WARRANTY for qualifying homeowners.

### B. Commercial / Business Use → 2 YEARS
- Warranty period: 2 years from original purchase date
- Applies to: Commercial installations, business properties, non-residential use

### C. Limited-Warranty Components → 1 YEAR ONLY
The following components are covered for 1 YEAR ONLY, regardless of residential or commercial use:
- Drain assemblies
- Bathroom hoses
- Hand-helds
- Kitchen hand sprays
- Kitchen hoses

⚠️ This list is EXPLICIT and CLOSED.
If a component is NOT listed above, it does NOT fall under the 1-year limitation.

---

## 3️⃣ CARTRIDGE COVERAGE CLARIFICATION

⚠️ CRITICAL: Cartridges are NOT listed under limited 1-year components.

Therefore:
- Residential homeowner → LIFETIME coverage
- Commercial use → 2-year coverage

❌ There is NO 1-year cartridge limitation.
❌ There is NO "extended warranty" structure.
❌ There is NO "standard 1-year warranty" for cartridges.

---

## 4️⃣ WARRANTY INTENT — REPAIR-FIRST POLICY

The primary objective of the warranty is:
- To RESTORE the product to proper working order
- NOT automatic full replacement

Warranty coverage provides:
- Replacement of defective PARTS
- Replacement of defective FINISHES
- Parts are provided FREE OF CHARGE during the applicable warranty period

Full product replacement:
- Is RARE
- Is determined SOLELY by Flusso
- Is NOT automatic

---

## 5️⃣ WARRANTY REQUIREMENTS

The following are REQUIRED for evaluation:
- ✅ Proof of purchase (invoice number is sufficient)
- ✅ Description of the issue
- ✅ Video evidence (PREFERRED)
- ✅ Photos only if the issue cannot be clearly demonstrated via video
- ✅ Completion of required troubleshooting steps (MANDATORY prior to approval)

---

## 6️⃣ INSTALLED PRODUCTS

Installed products:
- ❌ Are NOT eligible for return
- ✅ Must be handled STRICTLY through the warranty process

---

## 7️⃣ WARRANTY EXCLUSIONS

Warranty does NOT cover damage caused by:

### Installation & Usage
- Improper installation
- Product misuse or abuse
- Failure to use the product in its entirety

### Cleaning Damage
- Abrasive cleaners
- Alcohol-based cleaners
- Organic solvents

### Unauthorized Service
- Installation or servicing by unauthorized or unqualified individuals

---

## 8️⃣ THIRD-PARTY & UNAUTHORIZED PARTS

Under Magnuson-Moss compliance:
- Use of third-party parts does NOT automatically void the warranty

However:
- Failures caused by improperly installed, incompatible, or unauthorized components are NOT covered
- Affected portions of the warranty become inapplicable
- Use of genuine Flusso parts is STRONGLY recommended

---

## 9️⃣ UNAUTHORIZED SELLERS

Warranty does NOT apply to products purchased from unauthorized sellers.

---

## 🔟 COMPONENT-LEVEL WARRANTY RESOLUTION

Common repairable issues include:
- Cartridge issues
- Leaking hoses
- Finish defects on handles
- Aerator issues
- Loose spout, hoses, or handles

Standard resolution:
- Replace the DEFECTIVE COMPONENT ONLY
- Full unit replacement is RARE and internal determination only

---

## 11️⃣ AI ENFORCEMENT RULES (STRICT)

### The AI must NEVER:
- ❌ Invent a "standard 1-year full warranty"
- ❌ Mention a "2-year extended warranty"
- ❌ Create a "VIP warranty"
- ❌ Assign 2-year coverage to hoses
- ❌ Override limited component duration
- ❌ Default to full product replacement
- ❌ Ask for "original purchase date" when invoice number is provided

### ONLY valid durations in the system are:
- ✅ Lifetime (qualifying homeowner)
- ✅ 2 years (commercial use)
- ✅ 1 year (explicit limited components ONLY: drains, hoses, hand-helds, hand sprays)

⚠️ NO OTHER DURATIONS EXIST.

---

## RETURNS & REFUNDS

### Return Policy Timeline
| Days Since Purchase | Restocking Fee |
|---------------------|----------------|
| 0–45 days           | 15%            |
| 46–90 days          | 25%            |
| 91–180 days         | 50%            |
| 180+ days           | No returns accepted |

### Required for Returns
- ✅ Original packaging (unopened preferred, opened accepted with fee)
- ✅ RGA number (Return Goods Authorization) - MUST REQUEST FIRST
- ✅ Proof of purchase
- ❌ Custom/special orders are NOT returnable
- ❌ Clearance/final sale items are NOT returnable
- ❌ Installed products are NOT returnable

---

## MISSING PARTS (FROM NEW ORDERS)

### Policy
- Must report within 45 days of delivery
- Free replacement, no questions asked
- Need order number/PO number

### Process
1. Verify order number in system
2. Confirm what parts are missing (use packing list)
3. Ship missing parts immediately
4. No receipt needed - we have the order record

---

## ⚠️ MANDATORY REQUIREMENTS FOR WARRANTY CLAIMS

### BEFORE Processing ANY Warranty Claim:
1. ✅ **Proof of Purchase** - REQUIRED (invoice number is sufficient)
2. ✅ **Video/Photo of the Issue** - REQUIRED for defective products
3. ✅ **Shipping Address** - REQUIRED for replacements
4. ✅ **Troubleshooting Completion** - REQUIRED before approval

### NEVER DO THE FOLLOWING:
- ❌ DO NOT approve replacement without proof of purchase
- ❌ DO NOT ship replacements without confirmed shipping address
- ❌ DO NOT process warranty claims without photo/video evidence
- ❌ DO NOT assume information - always ask if not explicitly provided
- ❌ DO NOT invent warranty durations that don't exist
- ❌ DO NOT assume information - always ask if not explicitly provided
"""


# ===============================
# HELPERS
# ===============================
def _download_policy_doc() -> str:
    """Download policy document from Google Docs."""
    logger.info("[POLICY_SERVICE] Downloading policy document...")
    
    try:
        # Check if URL is configured
        if "YOUR_DOC_ID_HERE" in GOOGLE_DOCS_POLICY_URL:
            logger.warning("[POLICY_SERVICE] Google Docs URL not configured, using local fallback")
            return LOCAL_FALLBACK_POLICY
        
        response = requests.get(GOOGLE_DOCS_POLICY_URL, timeout=30)
        response.raise_for_status()
        
        text = response.text
        
        # Basic validation - should have some content
        if len(text) < 100:
            logger.warning("[POLICY_SERVICE] Downloaded content too short, using fallback")
            return LOCAL_FALLBACK_POLICY
        
        logger.info(f"[POLICY_SERVICE] Downloaded policy doc: {len(text)} characters")
        return text
        
    except Exception as e:
        logger.error(f"[POLICY_SERVICE] Error downloading policy: {e}")
        logger.info("[POLICY_SERVICE] Using local fallback policy")
        return LOCAL_FALLBACK_POLICY


def _parse_policy_sections(full_text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse the policy document into sections for quick lookup.
    Returns dict with section name -> {title, content, keywords}
    """
    sections = {}
    
    # Split by main headers (## 1. TITLE or # TITLE)
    # Pattern matches "## 1. WARRANTY" or "## WARRANTY" or "# WARRANTY"
    section_pattern = r'(?:^|\n)(#{1,2}\s*\d*\.?\s*([A-Z][A-Z\s&]+))\n'
    
    matches = list(re.finditer(section_pattern, full_text, re.MULTILINE))
    
    for i, match in enumerate(matches):
        section_title = match.group(2).strip()
        section_start = match.end()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        section_content = full_text[section_start:section_end].strip()
        
        # Normalize section name for lookup
        section_key = section_title.lower().replace(" ", "_").replace("&", "and")
        
        sections[section_key] = {
            "title": section_title,
            "content": section_content,
            "keywords": _extract_keywords(section_title)
        }
    
    # If parsing failed, create a single "general" section
    if not sections:
        sections["general"] = {
            "title": "General Policy",
            "content": full_text,
            "keywords": ["general", "policy"]
        }
    
    logger.info(f"[POLICY_SERVICE] Parsed {len(sections)} policy sections: {list(sections.keys())}")
    return sections


def _extract_keywords(title: str) -> List[str]:
    """Extract keywords from section title for matching."""
    title_lower = title.lower()
    keywords = []
    
    # Map titles to keywords
    keyword_map = {
        "warranty": ["warranty", "claim", "defect", "broken", "coverage"],
        "return": ["return", "refund", "rga", "money back"],
        "replacement": ["replacement", "part", "spare"],
        "missing": ["missing", "incomplete", "not included"],
        "escalation": ["escalate", "supervisor", "manager", "complex"],
        "vip": ["vip", "priority", "special"]
    }
    
    for key, kws in keyword_map.items():
        if key in title_lower:
            keywords.extend(kws)
    
    return keywords if keywords else [title_lower]


def _build_cache(full_text: str):
    """Build the policy cache from downloaded text."""
    global POLICY_CACHE, FULL_POLICY_TEXT, POLICY_SECTIONS
    
    logger.info("[POLICY_SERVICE] Building policy cache...")
    
    FULL_POLICY_TEXT = full_text
    POLICY_SECTIONS = _parse_policy_sections(full_text)
    
    # Build category -> section mapping
    new_cache = {}
    for section_key, section_data in POLICY_SECTIONS.items():
        new_cache[section_key] = section_data["content"]
    
    POLICY_CACHE = new_cache
    logger.info(f"[POLICY_SERVICE] Cache ready with {len(POLICY_CACHE)} sections")


def _refresh_cache():
    """Download and rebuild policy cache."""
    global LAST_REFRESH, IS_REFRESHING
    
    if IS_REFRESHING:
        return
    
    with LOCK:
        try:
            IS_REFRESHING = True
            text = _download_policy_doc()
            _build_cache(text)
            LAST_REFRESH = time.time()
            logger.info("[POLICY_SERVICE] Policy refresh complete")
        except Exception as e:
            logger.error(f"[POLICY_SERVICE] Refresh failed: {e}")
            # Ensure we have fallback
            if not FULL_POLICY_TEXT:
                _build_cache(LOCAL_FALLBACK_POLICY)
        finally:
            IS_REFRESHING = False


def _refresh_loop():
    """Background thread for periodic policy refresh."""
    logger.info("[POLICY_SERVICE] Background refresh thread started")
    while True:
        try:
            now = time.time()
            if now - LAST_REFRESH >= REFRESH_INTERVAL_SECONDS:
                logger.info("[POLICY_SERVICE] Starting scheduled refresh...")
                _refresh_cache()
            time.sleep(300)  # Check every 5 minutes
        except Exception as e:
            logger.error(f"[POLICY_SERVICE] Background loop error: {e}")
            time.sleep(300)


# ===============================
# PUBLIC API
# ===============================
def init_policy_service():
    """Initialize policy service on application startup."""
    logger.info("[POLICY_SERVICE] Initializing policy service...")
    _refresh_cache()
    
    # Start background refresh thread
    t = threading.Thread(target=_refresh_loop, daemon=True)
    t.start()


def get_full_policy() -> str:
    """Get the complete policy document."""
    if not FULL_POLICY_TEXT:
        _refresh_cache()
    return FULL_POLICY_TEXT


def get_policy_section(section_name: str) -> Optional[str]:
    """
    Get a specific policy section by name.
    
    Args:
        section_name: Section key like "warranty_claims", "returns_and_refunds"
    
    Returns:
        Section content or None if not found
    """
    if not POLICY_CACHE:
        _refresh_cache()
    
    # Normalize input
    normalized = section_name.lower().replace(" ", "_").replace("-", "_")
    
    # Direct match
    if normalized in POLICY_CACHE:
        return POLICY_CACHE[normalized]
    
    # Partial match
    for key, content in POLICY_CACHE.items():
        if normalized in key or key in normalized:
            return content
    
    return None


def get_relevant_policy(
    ticket_category: str = None,
    ticket_text: str = None,
    keywords: List[str] = None
) -> Dict[str, Any]:
    """
    Get relevant policy sections AND category tips based on ticket context.
    
    Args:
        ticket_category: Category from routing (e.g., "warranty_claim")
        ticket_text: Full ticket text for keyword matching
        keywords: Explicit keywords to search for
    
    Returns:
        {
            "primary_section": str,       # Most relevant section content
            "primary_section_name": str,  # Section name
            "additional_sections": [],    # Other potentially relevant sections
            "policy_requirements": [],    # Extracted requirements (what docs needed, etc.)
            "full_policy_available": bool,
            "category_tips": {            # LLM behavioral guidance (NEW)
                "label": str,
                "do": [],
                "dont": [],
                "required_fields": [],
                "sample_response": str or None
            }
        }
    """
    if not POLICY_SECTIONS:
        _refresh_cache()
    
    result = {
        "primary_section": "",
        "primary_section_name": "",
        "additional_sections": [],
        "policy_requirements": [],
        "full_policy_available": bool(FULL_POLICY_TEXT),
        "category_tips": None  # Will be populated from CATEGORY_TIPS
    }
    
    # Determine which categories to look for
    categories_to_check = []
    
    # From ticket category
    if ticket_category:
        cat_lower = ticket_category.lower()
        for policy_cat, keywords_list in POLICY_CATEGORIES.items():
            if cat_lower in keywords_list or any(k in cat_lower for k in keywords_list):
                categories_to_check.append(policy_cat)
    
    # From ticket text
    if ticket_text:
        text_lower = ticket_text.lower()
        for policy_cat, keywords_list in POLICY_CATEGORIES.items():
            if any(kw in text_lower for kw in keywords_list):
                if policy_cat not in categories_to_check:
                    categories_to_check.append(policy_cat)
    
    # From explicit keywords
    if keywords:
        for kw in keywords:
            kw_lower = kw.lower()
            for policy_cat, keywords_list in POLICY_CATEGORIES.items():
                if kw_lower in keywords_list or any(k in kw_lower for k in keywords_list):
                    if policy_cat not in categories_to_check:
                        categories_to_check.append(policy_cat)
    
    # Default to general if nothing matched
    if not categories_to_check:
        categories_to_check = ["general"]
    
    logger.info(f"[POLICY_SERVICE] Looking for policy sections: {categories_to_check}")
    
    # Find matching sections
    matched_sections = []
    for section_key, section_data in POLICY_SECTIONS.items():
        section_keywords = section_data.get("keywords", [])
        for cat in categories_to_check:
            if cat in section_key or any(cat in kw for kw in section_keywords):
                matched_sections.append({
                    "key": section_key,
                    "title": section_data["title"],
                    "content": section_data["content"]
                })
                break
    
    # Set primary and additional sections
    if matched_sections:
        result["primary_section"] = matched_sections[0]["content"]
        result["primary_section_name"] = matched_sections[0]["title"]
        result["additional_sections"] = [
            {"title": s["title"], "content": s["content"][:500] + "..."}
            for s in matched_sections[1:3]  # Max 2 additional
        ]
    else:
        # Fallback to full policy summary
        result["primary_section"] = FULL_POLICY_TEXT[:2000] if FULL_POLICY_TEXT else LOCAL_FALLBACK_POLICY[:2000]
        result["primary_section_name"] = "General Policy"
    
    # Extract requirements from primary section
    result["policy_requirements"] = _extract_requirements(result["primary_section"])
    
    # Get category tips for LLM behavioral guidance
    result["category_tips"] = _get_category_tips(ticket_category)
    
    return result


def _get_category_tips(ticket_category: str) -> Dict[str, Any]:
    """
    Get LLM behavioral tips for a ticket category.
    
    Args:
        ticket_category: Category from routing (e.g., "warranty_claim", "product_inquiry")
    
    Returns:
        Dict with label, do, dont, required_fields, sample_response
        Returns general tips if category not found.
    """
    if not ticket_category:
        return CATEGORY_TIPS.get("general", {})
    
    cat_lower = ticket_category.lower().strip()
    
    # Direct match first
    if cat_lower in CATEGORY_TIPS:
        return CATEGORY_TIPS[cat_lower]
    
    # Try mapping common variations
    category_mapping = {
        # warranty variations
        "warranty": "warranty_claim",
        "warranty_claim": "warranty_claim",
        # product issue variations
        "product_issue": "product_issue",
        "defect": "product_issue",
        # replacement variations
        "replacement_parts": "replacement_parts",
        "replacement": "replacement_parts",
        # missing parts variations
        "missing_parts": "missing_parts",
        "missing": "missing_parts",
        # return variations
        "return_refund": "return_refund",
        "return": "return_refund",
        "refund": "return_refund",
        # shipping variations
        "shipping_tracking": "shipping_tracking",
        "shipping": "shipping_tracking",
        "tracking": "shipping_tracking",
        # pricing variations
        "pricing_request": "pricing_request",
        "pricing": "pricing_request",
        # dealer variations
        "dealer_inquiry": "dealer_inquiry",
        "dealer": "dealer_inquiry",
        # installation variations
        "installation_help": "installation_help",
        "installation": "installation_help",
        # finish variations
        "finish_color": "finish_color",
        "finish": "finish_color",
        "color": "finish_color",
        # feedback variations
        "feedback_suggestion": "feedback_suggestion",
        "feedback": "feedback_suggestion",
        # product inquiry variations
        "product_inquiry": "product_inquiry",
        "stock_availability": "product_inquiry",
        # general
        "general": "general",
    }
    
    mapped_category = category_mapping.get(cat_lower, "general")
    return CATEGORY_TIPS.get(mapped_category, CATEGORY_TIPS.get("general", {}))


def format_category_tips_for_prompt(category_tips: Dict[str, Any], ticket_category: str = None) -> str:
    """
    Format category tips into a prompt section for the LLM.
    
    Args:
        category_tips: Dict from CATEGORY_TIPS with label, do, dont, etc.
        ticket_category: Original ticket category for display
    
    Returns:
        Formatted string to inject into LLM prompt
    """
    if not category_tips:
        return ""
    
    label = category_tips.get("label", ticket_category or "GENERAL")
    do_list = category_tips.get("do", [])
    dont_list = category_tips.get("dont", [])
    required_fields = category_tips.get("required_fields", [])
    sample_response = category_tips.get("sample_response")
    
    lines = []
    lines.append("═══════════════════════════════════════════════════════════════════════")
    lines.append(f"⚠️ CATEGORY: {label}")
    lines.append("═══════════════════════════════════════════════════════════════════════")
    lines.append("")
    
    # DO section
    if do_list:
        lines.append("✅ DO:")
        for item in do_list:
            lines.append(f"   • {item}")
        lines.append("")
    
    # DON'T section
    if dont_list:
        lines.append("❌ DO NOT:")
        for item in dont_list:
            lines.append(f"   • {item}")
        lines.append("")
    
    # Required fields
    if required_fields:
        lines.append("📋 REQUIRED BEFORE PROCESSING:")
        for field in required_fields:
            field_display = field.replace("_", " ").title()
            lines.append(f"   • {field_display}")
        lines.append("")
    
    # Sample response
    if sample_response:
        lines.append("📝 SAMPLE RESPONSE FORMAT:")
        lines.append(f'   "{sample_response}"')
        lines.append("")
    
    lines.append("═══════════════════════════════════════════════════════════════════════")
    
    return "\n".join(lines)


def _extract_requirements(policy_text: str) -> List[str]:
    """Extract key requirements from policy text (things marked with ✅ or "required")."""
    requirements = []
    
    # Look for checkmarks
    checkmark_pattern = r'[✅✓]\s*(.+?)(?:\n|$)'
    matches = re.findall(checkmark_pattern, policy_text)
    requirements.extend([m.strip() for m in matches])
    
    # Look for "required" or "must" statements
    required_pattern = r'(?:required|must|mandatory)[:\s]+(.+?)(?:\n|$)'
    matches = re.findall(required_pattern, policy_text, re.IGNORECASE)
    requirements.extend([m.strip() for m in matches])
    
    # Deduplicate and clean
    seen = set()
    unique_reqs = []
    for req in requirements:
        req_clean = req.strip().rstrip('.')
        if req_clean and req_clean.lower() not in seen:
            seen.add(req_clean.lower())
            unique_reqs.append(req_clean)
    
    return unique_reqs[:10]  # Max 10 requirements


def get_policy_for_category(category: str) -> str:
    """
    Simple helper to get policy text for a specific ticket category.
    
    Args:
        category: Ticket category like "warranty_claim", "return_refund", etc.
    
    Returns:
        Relevant policy text
    """
    result = get_relevant_policy(ticket_category=category)
    return result.get("primary_section", "")


# ===============================
# CONFIGURATION HELPER
# ===============================
def configure_policy_url(url: str):
    """
    Configure the Google Docs URL for policy document.
    Call this before init_policy_service() if you want to use a custom URL.
    
    Args:
        url: Google Docs export URL (use /export?format=txt)
    """
    global GOOGLE_DOCS_POLICY_URL
    GOOGLE_DOCS_POLICY_URL = url
    logger.info(f"[POLICY_SERVICE] Policy URL configured: {url[:50]}...")
