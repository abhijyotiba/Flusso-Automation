"""
Constraint Validator Service

This is the ENFORCEMENT layer that computes what fields are MISSING and what
policy citations MUST appear in responses. It bridges ticket_facts (what IS present)
with requirements_matrix (what SHOULD be present).

Key functions:
- validate_constraints(): Main entry point - computes all constraints
- format_constraints_for_prompt(): Formats constraints for LLM injection
- post_validate_response(): Checks if LLM response meets constraints

This module works WITH (not replaces) policy_service.py:
- policy_service.py provides full policy TEXT for LLM context
- This module provides specific CONSTRAINTS for enforcement
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from app.config.requirements_matrix import (
    REQUIREMENTS_MATRIX,
    CATEGORY_ALIASES,
    FIELD_NAMES,
    FIELD_TO_FACTS_KEY,
    FIELD_ASK_TEMPLATES,
    get_canonical_category,
    get_requirements_for_category,
    is_strictly_defined_category,
)
from app.config.policy_rules import (
    POLICY_RULES,
    POLICY_TRIGGERS,
    get_policy_rule,
    get_policies_for_product,
    get_policies_for_category,
    get_policy_citation,
    get_all_policy_citations,
)

logger = logging.getLogger(__name__)
STEP_NAME = "🔒 CONSTRAINT_VALIDATOR"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ConstraintResult:
    """Result of constraint validation."""
    # Category info
    original_category: str
    resolved_category: str
    
    # Missing fields (what to ask for)
    missing_fields: List[str] = field(default_factory=list)
    required_asks: List[str] = field(default_factory=list)
    
    # Present fields (what NOT to ask for)
    present_fields: List[str] = field(default_factory=list)
    must_not_ask: List[str] = field(default_factory=list)
    
    # Policy constraints
    applicable_policies: List[str] = field(default_factory=list)
    policy_citations: List[Dict[str, str]] = field(default_factory=list)
    required_citations: List[str] = field(default_factory=list)
    
    # Conditional info (may need based on context)
    conditional_fields: Dict[str, str] = field(default_factory=dict)
    
    # Validation flags
    can_proceed: bool = True
    blocking_missing: List[str] = field(default_factory=list)
    
    # Skip flag - True when category is not in strict validation list
    skipped: bool = False
    
    # Metadata
    validation_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "original_category": self.original_category,
            "resolved_category": self.resolved_category,
            "missing_fields": self.missing_fields,
            "required_asks": self.required_asks,
            "present_fields": self.present_fields,
            "must_not_ask": self.must_not_ask,
            "applicable_policies": self.applicable_policies,
            "policy_citations": self.policy_citations,
            "required_citations": self.required_citations,
            "conditional_fields": self.conditional_fields,
            "can_proceed": self.can_proceed,
            "blocking_missing": self.blocking_missing,
            "skipped": self.skipped,
            "validation_notes": self.validation_notes,
        }


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def validate_constraints(
    ticket_facts: Dict[str, Any],
    ticket_category: str,
    product_text: Optional[str] = None,
) -> ConstraintResult:
    """
    Main entry point - validates ticket against requirements and policies.
    
    This function:
    1. Checks if category is in STRICT_VALIDATION_CATEGORIES
    2. If NOT strictly defined → returns skipped result (no enforcement)
    3. If strictly defined → full validation with missing/present fields
    
    Args:
        ticket_facts: From ticket_extractor (has_address, has_receipt, etc.)
        ticket_category: From routing (warranty_claim, missing_parts, etc.)
        product_text: Optional product description for policy matching
        
    Returns:
        ConstraintResult with all computed constraints (or skipped=True for flexible categories)
    """
    logger.info(f"{STEP_NAME} | Validating constraints for category: {ticket_category}")
    
    # Initialize result
    result = ConstraintResult(
        original_category=ticket_category or "unknown",
        resolved_category="general"
    )
    
    # 1. Resolve category
    resolved_category = get_canonical_category(ticket_category)
    result.resolved_category = resolved_category
    logger.info(f"{STEP_NAME} | Resolved category: {ticket_category} → {resolved_category}")
    
    # 2. CHECK IF CATEGORY REQUIRES STRICT VALIDATION
    # Only categories in STRICT_VALIDATION_CATEGORIES get enforced
    if not is_strictly_defined_category(ticket_category):
        # Skip validation for non-strict categories (process as general ticket)
        result.skipped = True
        result.can_proceed = True
        result.validation_notes.append(f"Category '{resolved_category}' is not in strict validation list - processing flexibly")
        logger.info(f"{STEP_NAME} | ⏭️ SKIPPED - Category '{resolved_category}' not in strict validation list")
        logger.info(f"{STEP_NAME} | Processing as flexible/general ticket without field enforcement")
        return result
    
    logger.info(f"{STEP_NAME} | ✅ Category '{resolved_category}' requires strict validation")
    
    # 3. Get requirements for category (strict validation)
    requirements = get_requirements_for_category(resolved_category)
    required_fields = requirements.get("required", [])
    conditional_fields = requirements.get("conditional", {})
    policy_keys = requirements.get("policies", [])
    product_policies = requirements.get("product_specific_policies", {})
    
    # 4. Compute missing vs present fields
    missing, present = _compute_field_status(ticket_facts, required_fields)
    result.missing_fields = missing
    result.present_fields = present
    
    # 5. Generate ask messages for missing fields
    result.required_asks = _generate_ask_messages(missing)
    
    # 6. Generate must-not-ask list (fields that are present)
    result.must_not_ask = _generate_must_not_ask(present, ticket_facts)
    
    # 6. Handle conditional fields
    result.conditional_fields = _evaluate_conditional_fields(
        conditional_fields, ticket_facts, resolved_category
    )
    
    # 6b. Handle claimed-but-missing attachments (discrepancy detection)
    # If customer CLAIMED to attach something but it wasn't received, add to missing_fields
    claimed_but_missing = ticket_facts.get("claimed_but_missing", []) or []
    if claimed_but_missing and resolved_category in ["product_issue", "warranty_claim", "replacement_parts", "return_refund"]:
        for attachment_type in claimed_but_missing:
            if attachment_type == "video" and "video" not in result.missing_fields:
                result.missing_fields.append("video")
                result.required_asks.append(
                    "Thank you for sending over the video. If the file is larger than 20MB or if the file was sent through a Google Drive link, "
                    "we may not be able to access it. One option you have is sending it through wetransfer.com and sharing the download link with us."
                )
                result.validation_notes.append(f"Customer claimed to attach {attachment_type} but it was not received (possibly >20MB limit)")
            elif attachment_type == "photos" and "photos" not in result.missing_fields:
                result.missing_fields.append("photos")
                result.required_asks.append(
                    "We noticed you mentioned attaching photos/images, but they don't appear to have come through. "
                    "Could you please re-attach the photos showing the issue?"
                )
                result.validation_notes.append(f"Customer claimed to attach {attachment_type} but they were not received")
            elif attachment_type == "documents" and "documents" not in result.missing_fields:
                result.missing_fields.append("documents")
                result.required_asks.append(
                    "We noticed you mentioned attaching a document, but it doesn't appear to have come through. "
                    "Could you please re-attach the document?"
                )
                result.validation_notes.append(f"Customer claimed to attach {attachment_type} but they were not received")
        
        if claimed_but_missing:
            logger.warning(f"{STEP_NAME} | 🚨 Attachment discrepancy: customer claimed {claimed_but_missing} but not received")
    
    # 7. Determine applicable policies
    all_policies = list(policy_keys)  # Start with category policies
    
    # Add product-specific policies
    if product_text:
        product_policies_list = get_policies_for_product(product_text)
        for p in product_policies_list:
            if p not in all_policies:
                all_policies.append(p)
    
    # Check ticket text for product keywords
    ticket_text = ticket_facts.get("ticket_text_snippet", "") or ""
    raw_products = ticket_facts.get("raw_product_codes", [])
    for code_info in raw_products:
        model = code_info.get("model", "") or ""
        # Check if model contains hose-related keywords
        for keyword, policy in product_policies.items():
            if keyword in model.lower() or keyword in ticket_text.lower():
                if policy not in all_policies:
                    all_policies.append(policy)
    
    result.applicable_policies = all_policies
    
    # 8. Get policy citations
    result.policy_citations = get_all_policy_citations(all_policies)
    result.required_citations = [c["citation"] for c in result.policy_citations if c.get("citation")]
    
    # 9. Determine if we can proceed or need blocking info
    # Some missing fields are blocking (can't process without them)
    blocking_fields = _get_blocking_fields(resolved_category, missing)
    result.blocking_missing = blocking_fields
    result.can_proceed = len(blocking_fields) == 0
    
    # 10. Add validation notes
    if missing:
        result.validation_notes.append(f"Missing required fields: {', '.join(missing)}")
    if result.required_citations:
        result.validation_notes.append(f"Must cite {len(result.required_citations)} policy(ies)")
    if not result.can_proceed:
        result.validation_notes.append(f"Blocking missing: {', '.join(blocking_fields)}")
    
    logger.info(f"{STEP_NAME} | ✅ Validation complete:")
    logger.info(f"{STEP_NAME} |   Missing: {missing}")
    logger.info(f"{STEP_NAME} |   Present: {present}")
    logger.info(f"{STEP_NAME} |   Policies: {all_policies}")
    logger.info(f"{STEP_NAME} |   Can proceed: {result.can_proceed}")
    
    return result


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _compute_field_status(
    ticket_facts: Dict[str, Any],
    required_fields: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Compute which required fields are missing vs present.
    
    Args:
        ticket_facts: Ticket facts dict
        required_fields: List of required field keys
        
    Returns:
        Tuple of (missing_fields, present_fields)
    """
    if not ticket_facts:
        ticket_facts = {}
    
    missing = []
    present = []
    
    for field_key in required_fields:
        facts_key = FIELD_TO_FACTS_KEY.get(field_key, f"has_{field_key}")
        
        # Check if field is present in ticket_facts
        value = ticket_facts.get(facts_key)
        
        # Handle different value types
        if isinstance(value, bool):
            is_present = value
        elif isinstance(value, list):
            is_present = len(value) > 0
        elif isinstance(value, str):
            is_present = bool(value.strip())
        else:
            is_present = value is not None
        
        if is_present:
            present.append(field_key)
        else:
            missing.append(field_key)
    
    return missing, present


def _generate_ask_messages(missing_fields: List[str]) -> List[str]:
    """
    Generate customer-friendly ask messages for missing fields.
    
    Args:
        missing_fields: List of missing field keys
        
    Returns:
        List of ask message strings
    """
    messages = []
    for field_key in missing_fields:
        template = FIELD_ASK_TEMPLATES.get(field_key)
        if template:
            messages.append(template)
        else:
            # Fallback generic message
            field_name = FIELD_NAMES.get(field_key, field_key.replace("_", " "))
            messages.append(f"Could you please provide the {field_name}?")
    return messages


def _generate_must_not_ask(
    present_fields: List[str],
    ticket_facts: Dict[str, Any]
) -> List[str]:
    """
    Generate list of things NOT to ask for (already present).
    
    Args:
        present_fields: List of present field keys
        ticket_facts: Full ticket facts for additional context
        
    Returns:
        List of must-not-ask items with context
    """
    must_not_ask = []
    
    for field_key in present_fields:
        field_name = FIELD_NAMES.get(field_key, field_key.replace("_", " "))
        must_not_ask.append(field_name)
    
    # Add specific items based on ticket_facts
    if ticket_facts:
        # If we have model numbers, don't ask for model
        if ticket_facts.get("raw_product_codes") or ticket_facts.get("has_model_number"):
            models = ticket_facts.get("raw_product_codes", [])
            if models:
                model_str = ", ".join([m.get("model", "") for m in models[:2]])
                if "model" not in must_not_ask:
                    must_not_ask.append(f"product model (already provided: {model_str})")
        
        # If we have an address, don't ask for address
        if ticket_facts.get("extracted_address"):
            addr = ticket_facts.get("extracted_address", "")[:30]
            if "address" not in str(must_not_ask):
                must_not_ask.append(f"shipping address (already provided: {addr}...)")
        
        # If we have photos, don't ask for photos
        if ticket_facts.get("has_photos"):
            if "photo" not in str(must_not_ask).lower():
                must_not_ask.append("photos (already attached)")
        
        # If we have finish info, don't ask for finish
        if ticket_facts.get("raw_finish_mentions"):
            finishes = ticket_facts.get("raw_finish_mentions", [])
            if finishes and "finish" not in str(must_not_ask).lower():
                must_not_ask.append(f"finish/color (mentioned: {', '.join(finishes[:2])})")
    
    return must_not_ask


def _evaluate_conditional_fields(
    conditional_fields: Dict[str, Dict[str, str]],
    ticket_facts: Dict[str, Any],
    category: str
) -> Dict[str, str]:
    """
    Evaluate which conditional fields might be needed.
    
    Args:
        conditional_fields: Dict of conditional field definitions
        ticket_facts: Ticket facts
        category: Resolved category
        
    Returns:
        Dict of field_key -> reason (for fields that might be needed)
    """
    might_need = {}
    
    for field_key, condition_info in conditional_fields.items():
        condition = condition_info.get("condition", "")
        reason = condition_info.get("reason", "")
        
        # Check if condition applies
        should_include = False
        
        if condition == "always_for_defect":
            # Photos always needed for defect claims
            if category in ["warranty_claim", "product_issue"]:
                should_include = True
                
        elif condition == "warranty_check_needed":
            # Receipt needed if warranty replacement possible
            if category in ["product_issue", "replacement_parts"]:
                should_include = True
                
        elif condition == "replacement_offered":
            # Address needed if sending replacement
            should_include = True
            
        elif condition == "intermittent_issue":
            # Video for intermittent problems (can't auto-detect, suggest if no photos)
            if not ticket_facts.get("has_photos") and not ticket_facts.get("has_video"):
                should_include = True
        
        # Check if field already present
        facts_key = FIELD_TO_FACTS_KEY.get(field_key, f"has_{field_key}")
        is_present = ticket_facts.get(facts_key, False)
        
        if should_include and not is_present:
            might_need[field_key] = reason
    
    return might_need


def _get_blocking_fields(category: str, missing_fields: List[str]) -> List[str]:
    """
    Determine which missing fields are blocking (can't proceed without).
    
    Args:
        category: Ticket category
        missing_fields: List of missing field keys
        
    Returns:
        List of blocking field keys
    """
    # Define which fields are blocking per category
    blocking_rules = {
        "warranty_claim": ["receipt"],  # Must have proof of purchase
        "return_refund": ["receipt"],   # Must have proof of purchase
        "missing_parts": ["po"],        # Must have order number
        "shipping_tracking": ["po"],    # Must have order number
    }
    
    blocking_for_category = blocking_rules.get(category, [])
    blocking = [f for f in missing_fields if f in blocking_for_category]
    
    return blocking


# =============================================================================
# PROMPT FORMATTING
# =============================================================================

def format_constraints_for_prompt(result: ConstraintResult | Dict[str, Any]) -> str:
    """
    Format constraints for injection into LLM prompt.
    
    Args:
        result: ConstraintResult dataclass OR dict (from .to_dict())
        
    Returns:
        Formatted string for prompt injection (empty if skipped)
    """
    # Handle both dataclass and dict input
    if isinstance(result, dict):
        skipped = result.get("skipped", False)
        must_not_ask = result.get("must_not_ask", [])
        required_asks = result.get("required_asks", [])
        conditional_fields = result.get("conditional_fields", {})
        required_citations = result.get("required_citations", [])
        can_proceed = result.get("can_proceed", True)
        blocking_missing = result.get("blocking_missing", [])
    else:
        skipped = result.skipped
        must_not_ask = result.must_not_ask
        required_asks = result.required_asks
        conditional_fields = result.conditional_fields
        required_citations = result.required_citations
        can_proceed = result.can_proceed
        blocking_missing = result.blocking_missing
    
    # If validation was skipped, return empty string (no constraints to enforce)
    if skipped:
        return ""
    
    lines = [
        "",
        "═══════════════════════════════════════════════════════════════════════",
        "🔒 MANDATORY CONSTRAINTS - YOU MUST FOLLOW THESE",
        "═══════════════════════════════════════════════════════════════════════",
        ""
    ]
    
    # Must NOT ask section
    if must_not_ask:
        lines.append("❌ DO NOT ask for the following (already provided):")
        for item in must_not_ask:
            lines.append(f"   • {item}")
        lines.append("")
    
    # MUST ask section
    if required_asks:
        lines.append("✅ YOU MUST ask for the following (missing required info):")
        for ask in required_asks:
            lines.append(f"   • {ask}")
        lines.append("")
    
    # Conditional (might need)
    if conditional_fields:
        lines.append("⚠️ CONSIDER asking for (depending on context):")
        for field_key, reason in conditional_fields.items():
            field_name = FIELD_NAMES.get(field_key, field_key)
            lines.append(f"   • {field_name} - {reason}")
        lines.append("")
    
    # Policy citations
    if required_citations:
        lines.append("📜 YOU MUST include these policy statements in your response:")
        for citation in required_citations:
            lines.append(f"   • \"{citation}\"")
        lines.append("")
    
    # Blocking info warning
    if not can_proceed and blocking_missing:
        lines.append("⛔ BLOCKING: Cannot process request without:")
        for field in blocking_missing:
            field_name = FIELD_NAMES.get(field, field)
            lines.append(f"   • {field_name}")
        lines.append("   → You MUST request this information before proceeding.")
        lines.append("")
    
    lines.append("═══════════════════════════════════════════════════════════════════════")
    lines.append("")
    
    return "\n".join(lines)


def format_constraints_summary(result: ConstraintResult | Dict[str, Any]) -> str:
    """
    Format a brief summary of constraints for logging/display.
    
    Args:
        result: ConstraintResult dataclass OR dict
        
    Returns:
        Brief summary string
    """
    # Handle both dataclass and dict input
    if isinstance(result, dict):
        skipped = result.get("skipped", False)
        missing_fields = result.get("missing_fields", [])
        must_not_ask = result.get("must_not_ask", [])
        applicable_policies = result.get("applicable_policies", [])
        can_proceed = result.get("can_proceed", True)
        blocking_missing = result.get("blocking_missing", [])
        resolved_category = result.get("resolved_category", "")
    else:
        skipped = result.skipped
        missing_fields = result.missing_fields
        must_not_ask = result.must_not_ask
        applicable_policies = result.applicable_policies
        can_proceed = result.can_proceed
        blocking_missing = result.blocking_missing
        resolved_category = result.resolved_category
    
    # If skipped, return clear message
    if skipped:
        return f"SKIPPED ('{resolved_category}' not in strict list) - processing flexibly"
    
    parts = []
    
    if missing_fields:
        parts.append(f"Missing: {', '.join(missing_fields)}")
    
    if must_not_ask:
        # Just list field names, not full descriptions
        fields = [f.split(" (")[0] for f in must_not_ask[:3]]
        parts.append(f"Present: {', '.join(fields)}")
    
    if applicable_policies:
        parts.append(f"Policies: {', '.join(applicable_policies)}")
    
    if not can_proceed:
        parts.append(f"BLOCKED by: {', '.join(blocking_missing)}")
    
    return " | ".join(parts) if parts else "No constraints"


# =============================================================================
# POST-VALIDATION (CHECK LLM RESPONSE)
# =============================================================================

def post_validate_response(
    response_text: str,
    constraints: ConstraintResult | Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check if LLM response meets the constraints.
    
    Args:
        response_text: Generated response text
        constraints: ConstraintResult dataclass OR dict (from .to_dict())
        
    Returns:
        {
            "valid": bool,  # alias for is_valid
            "is_valid": bool,
            "violations": List[str],
            "missing_citations": List[str],
            "unnecessary_asks": List[str],
            "suggestions": List[str],
            "warnings": List[str],  # Combined warnings for convenience
            "skipped": bool,  # True if validation was skipped
        }
    """
    # Check if constraints were skipped (non-strict category)
    if isinstance(constraints, dict):
        skipped = constraints.get("skipped", False)
    else:
        skipped = constraints.skipped
    
    # If constraints were skipped, return valid without checking
    if skipped:
        return {
            "valid": True,
            "is_valid": True,
            "violations": [],
            "missing_citations": [],
            "unnecessary_asks": [],
            "suggestions": [],
            "warnings": [],
            "skipped": True,
        }
    
    violations = []
    missing_citations = []
    unnecessary_asks = []
    suggestions = []
    
    response_lower = response_text.lower()
    
    # Handle both dataclass and dict input
    if isinstance(constraints, dict):
        required_citations = constraints.get("required_citations", [])
        must_not_ask = constraints.get("must_not_ask", [])
        required_asks = constraints.get("required_asks", [])
    else:
        required_citations = constraints.required_citations
        must_not_ask = constraints.must_not_ask
        required_asks = constraints.required_asks
    
    # Check for required citations
    for citation in required_citations:
        # Check if key parts of citation are present
        # We do fuzzy matching - check for key numbers/terms
        citation_lower = citation.lower()
        
        # Extract key terms to check
        key_terms = []
        
        # Check for time periods (e.g., "1 year", "2 years", "45 days")
        time_matches = re.findall(r'\d+\s*(?:year|month|day)s?', citation_lower)
        key_terms.extend(time_matches)
        
        # Check for key policy words
        policy_words = ["warranty", "return", "missing parts", "restocking"]
        for word in policy_words:
            if word in citation_lower:
                key_terms.append(word)
        
        # Verify at least some key terms are in response
        terms_found = sum(1 for term in key_terms if term in response_lower)
        
        if key_terms and terms_found < len(key_terms) / 2:
            missing_citations.append(citation)
    
    # Check for unnecessary asks (asking for things already provided)
    ask_patterns = [
        (r"provide.*(?:model|model number)", "model"),
        (r"(?:what|which).*model", "model"),
        (r"provide.*address", "address"),
        (r"(?:what|which).*address", "address"),
        (r"send.*(?:photo|picture|image)", "photos"),
        (r"provide.*(?:receipt|invoice|proof)", "receipt"),
        (r"(?:what|which).*finish|color", "finish"),
    ]
    
    for pattern, field in ask_patterns:
        if re.search(pattern, response_lower):
            # Check if this field is in must_not_ask
            for must_not in must_not_ask:
                if field in must_not.lower():
                    unnecessary_asks.append(f"Asked for {field} which was already provided")
                    break
    
    # Check for missing required asks
    for ask in required_asks:
        # Extract the key field being asked for
        ask_lower = ask.lower()
        
        asked_in_response = False
        
        # Check if the ask topic appears in response
        if "address" in ask_lower and any(w in response_lower for w in ["address", "where", "ship"]):
            asked_in_response = True
        elif "receipt" in ask_lower and any(w in response_lower for w in ["receipt", "invoice", "proof of purchase", "purchase date"]):
            asked_in_response = True
        elif "photo" in ask_lower and any(w in response_lower for w in ["photo", "picture", "image", "send us"]):
            asked_in_response = True
        elif "model" in ask_lower and any(w in response_lower for w in ["model", "product number"]):
            asked_in_response = True
        elif "po" in ask_lower or "order" in ask_lower:
            if any(w in response_lower for w in ["order number", "po number", "purchase order", "order confirmation"]):
                asked_in_response = True
        
        if not asked_in_response:
            violations.append(f"Did not ask for required info: {ask[:50]}...")
    
    # Generate suggestions
    if missing_citations:
        suggestions.append(f"Add missing policy citation(s): {len(missing_citations)} citation(s) not found")
    
    if unnecessary_asks:
        suggestions.append(f"Remove unnecessary asks: {', '.join([a.split(' which')[0] for a in unnecessary_asks])}")
    
    is_valid = len(violations) == 0 and len(missing_citations) == 0 and len(unnecessary_asks) == 0
    
    # Build combined warnings list for convenience
    warnings = violations + [f"Missing citation: {c[:50]}..." for c in missing_citations] + unnecessary_asks
    
    return {
        "valid": is_valid,  # Alias for compatibility
        "is_valid": is_valid,
        "violations": violations,
        "missing_citations": missing_citations,
        "unnecessary_asks": unnecessary_asks,
        "suggestions": suggestions,
        "warnings": warnings,
        "skipped": False,
    }


def enforce_constraints_on_response(
    response_text: str,
    constraints: ConstraintResult | Dict[str, Any]
) -> str:
    """
    Modify response to enforce constraints (add missing citations, etc.)
    
    Args:
        response_text: Original response
        constraints: ConstraintResult dataclass OR dict (from .to_dict())
        
    Returns:
        Modified response text
    """
    validation = post_validate_response(response_text, constraints)
    
    if validation["is_valid"]:
        return response_text
    
    modified = response_text
    additions = []
    
    # Handle both dataclass and dict input for required_asks
    if isinstance(constraints, dict):
        required_asks = constraints.get("required_asks", [])
    else:
        required_asks = constraints.required_asks
    
    # Add missing citations
    if validation["missing_citations"]:
        additions.append("\n\n**Policy Information:**")
        for citation in validation["missing_citations"]:
            additions.append(f"• {citation}")
    
    # Add missing required asks
    if validation["violations"]:
        # Check if we need to add asks
        missing_ask_topics = []
        for violation in validation["violations"]:
            if "Did not ask for" in violation:
                missing_ask_topics.append(violation)
        
        if missing_ask_topics:
            additions.append("\n\n**To help us assist you better, please provide:**")
            for ask in required_asks:
                # Add asks that weren't in response
                additions.append(f"• {ask}")
    
    if additions:
        modified = response_text.rstrip() + "\n" + "\n".join(additions)
    
    return modified


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_constraints_for_ticket(
    ticket_facts: Dict[str, Any],
    ticket_category: str,
    product_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function that returns constraints as a dict.
    
    Args:
        ticket_facts: Ticket facts
        ticket_category: Ticket category
        product_text: Optional product text
        
    Returns:
        Dict with constraint information
    """
    result = validate_constraints(ticket_facts, ticket_category, product_text)
    return result.to_dict()
