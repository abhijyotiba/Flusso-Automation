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
    "replacement_parts": ["replacement", "replacement_parts", "spare part", "part needed", "missing_parts"],
    "missing_parts": ["missing", "missing_parts", "not included", "incomplete", "parts missing"],
    "shipping": ["shipping", "shipping_tracking", "delivery", "tracking", "where is my order"],
    "installation": ["installation", "installation_help", "install", "setup", "mounting", "how to"],
    "general": ["general", "product_inquiry", "question", "information"]
}

# ===============================
# LOCAL FALLBACK POLICY
# ===============================
LOCAL_FALLBACK_POLICY = """
# Flusso Kitchen & Bath - Policy & Process Guide

## 1. WARRANTY CLAIMS

### 1.1 Warranty Coverage
- Standard warranty: 1 year from purchase date
- Extended warranty (VIP customers): 2 years
- Lifetime warranty products: Check product specifications

### 1.2 Required Documents for Warranty Claim
- ✅ Proof of purchase (receipt/invoice) - MANDATORY
- ✅ Product model number
- ✅ Photos of defect (recommended)
- ✅ Purchase date must be within warranty period

### 1.3 Warranty Process
1. Verify product model and purchase date
2. Confirm defect is covered (not misuse/normal wear)
3. If approved: Send replacement part OR full product replacement
4. If denied: Explain reason clearly, offer paid alternatives

### 1.4 NOT Covered Under Warranty
- Cosmetic damage (scratches, discoloration from cleaning products)
- Normal wear and tear
- Damage from improper installation
- Products without valid proof of purchase
- Products purchased from unauthorized dealers

---

## 2. RETURNS & REFUNDS

### 2.1 Return Policy Timeline
| Days Since Purchase | Restocking Fee |
|---------------------|----------------|
| 0-45 days           | 15%            |
| 46-90 days          | 25%            |
| 91-180 days         | 50%            |
| 180+ days           | No returns accepted |

### 2.2 Required for Returns
- ✅ Original packaging (unopened preferred, opened accepted with fee)
- ✅ RGA number (Return Goods Authorization) - MUST REQUEST FIRST
- ✅ Proof of purchase with date
- ❌ Custom/special orders are NOT returnable
- ❌ Clearance/final sale items are NOT returnable

### 2.3 Return Process
1. Customer requests RGA via email/ticket
2. Verify purchase date and calculate applicable fee
3. Issue RGA number with return instructions
4. Customer ships product back (customer pays return shipping)
5. Receive and inspect product
6. Process refund minus restocking fee within 5-7 business days

---

## 3. REPLACEMENT PARTS

### 3.1 Free Replacement Parts (Under Warranty)
- Must be within warranty period
- Must have valid proof of purchase
- Only for defective parts (not lost/damaged by user)
- We ship free, customer does not pay

### 3.2 Paid Replacement Parts (Out of Warranty)
- Available for all products regardless of purchase date
- Customer pays part cost + shipping
- No proof of purchase required
- Check availability before promising

### 3.3 Process for Replacement Parts
1. Identify product model number accurately
2. Identify specific part needed (use parts diagrams if available)
3. Check warranty status with proof of purchase
4. If under warranty: Ship replacement free
5. If not under warranty: Quote price, get approval, process

---

## 4. MISSING PARTS (FROM NEW ORDERS)

### 4.1 Policy
- Must report within 30 days of delivery
- Free replacement, no questions asked
- Need order number/PO number

### 4.2 Process
1. Verify order number in system
2. Confirm what parts are missing (use packing list)
3. Ship missing parts immediately via expedited shipping
4. No receipt needed - we have the order record
5. Apologize for inconvenience

---

## 5. ESCALATION RULES

### 5.1 When to Escalate to Human Agent
- Customer disputes warranty decision
- Damage claims over $500 value
- Legal threats or lawyer mentions
- VIP customer complaints (flag for priority)
- Unusual/complex multi-product situations
- Customer requests supervisor/manager

### 5.2 VIP Customer Special Rules
- Always approve reasonable warranty claims
- Free shipping on ALL replacements (even paid parts)
- Extended return window: 90 days full refund (no restocking fee)
- Priority response required (same day if possible)
- Can override standard policies with manager approval note
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
    Get relevant policy sections based on ticket context.
    
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
            "full_policy_available": bool
        }
    """
    if not POLICY_SECTIONS:
        _refresh_cache()
    
    result = {
        "primary_section": "",
        "primary_section_name": "",
        "additional_sections": [],
        "policy_requirements": [],
        "full_policy_available": bool(FULL_POLICY_TEXT)
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
    
    return result


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
