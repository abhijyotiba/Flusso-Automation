# Constraint Validator Implementation

## Executive Summary

The **Constraint Validator** is a unified enforcement layer that ensures customer support responses include all required information requests and policy citations. It bridges the gap between ticket fact extraction and response generation by providing deterministic, rule-based constraints that the LLM must follow.

---

## Table of Contents

1. [Background: Previous Implementation](#background-previous-implementation)
2. [Identified Gaps](#identified-gaps)
3. [Solution: Unified Constraint Validator](#solution-unified-constraint-validator)
4. [Architecture Overview](#architecture-overview)
5. [Components](#components)
6. [Data Flow](#data-flow)
7. [Technical Implementation](#technical-implementation)
8. [Integration Points](#integration-points)
9. [Configuration Guide](#configuration-guide)
10. [Testing](#testing)

---

## Background: Previous Implementation

### What Existed Before

The Flusso workflow automation system already had several components for handling customer support tickets:

#### 1. **Ticket Extractor** (`app/nodes/ticket_extractor.py`)
- Extracted structured facts from ticket text using LLM
- Identified: product codes, purchase dates, customer contact info, issue descriptions
- Output: `ticket_facts` dictionary with boolean flags like `has_address`, `receipt_provided`

#### 2. **Policy Service** (`app/services/policy_service.py`)
- Retrieved relevant policy TEXT from Google Docs
- Provided full policy sections for LLM context
- Output: Complete policy documents for categories (warranty, returns, shipping)

#### 3. **Routing Agent** (`app/nodes/routing_agent.py`)
- Classified tickets into categories: `warranty_claim`, `missing_parts`, `return_refund`, etc.
- Determined workflow routing based on ticket content

#### 4. **React Agent** (`app/nodes/react_agent.py`)
- Main reasoning agent using ReAct pattern
- Used tools to search documents, analyze products, look up past tickets
- Generated responses based on gathered context

#### 5. **Draft Response** (`app/nodes/response/draft_response.py`)
- Generated final customer-facing responses
- Added confidence scores and source citations
- Basic missing requirements detection (hardcoded checks)

### How It Worked (Pre-Implementation)

```
Ticket → Extraction → Routing → Planning → Agent → Response
           ↓            ↓
      ticket_facts  category
           ↓            ↓
      (disconnected from enforcement)
```

The system extracted facts and routed tickets, but **the extracted facts were not systematically enforced** in the response generation phase.

---

## Identified Gaps

### Gap 1: No Field Requirements Matrix

**Problem:** There was no centralized definition of what information is REQUIRED for each ticket category.

| Scenario | What Happened |
|----------|---------------|
| Warranty claim without receipt | Agent might not ask for receipt |
| Missing parts claim without PO | Agent inconsistently requested PO |
| Return request without address | Sometimes asked, sometimes didn't |

**Root Cause:** Requirements were scattered across prompts or hardcoded in draft_response.py, not systematically defined.

### Gap 2: Extraction Without Enforcement

**Problem:** `ticket_extractor` correctly identified `receipt_provided: false`, but this information wasn't used to FORCE the response to ask for a receipt.

```python
# ticket_facts from extractor
{
    "receipt_provided": False,  # ← Extracted correctly
    "shipping_address_provided": False,
    "po_number": None,
}

# But response might still say:
"We'll process your warranty claim right away!"  # ← No ask for missing items!
```

**Root Cause:** No mechanism to compute `missing_fields = required - present` and inject this into the LLM prompt as a hard constraint.

### Gap 3: Policy TEXT vs Policy RULES

**Problem:** `policy_service.py` provided full policy TEXT, but the agent treated it as context/suggestions rather than mandatory requirements.

| What Policy Service Provided | What Agent Did |
|------------------------------|----------------|
| "Warranty period is 12 months from purchase" | Sometimes mentioned, sometimes didn't |
| "Missing parts must be reported within 45 days" | Inconsistently cited |
| "Returns subject to 15% restocking fee" | Often omitted |

**Root Cause:** Full text is good for understanding, but specific CITATIONS need to be extracted and marked as MUST-INCLUDE.

### Gap 4: No Post-Validation

**Problem:** Even with good prompts, LLMs can still:
- Forget to ask for required items
- Omit policy citations
- Ask for information already provided

**Root Cause:** No validation layer to check the generated response against constraints before sending.

### Gap 5: "Must Not Ask" Logic Missing

**Problem:** When a customer provided their address, the agent sometimes still asked for it.

```
Customer: "Please ship to 123 Main St, City, ST 12345"
Agent: "Could you please provide your shipping address?"  # ← Annoying!
```

**Root Cause:** No tracking of what's ALREADY PROVIDED to prevent redundant asks.

---

## Solution: Unified Constraint Validator

### Design Philosophy

Instead of two separate implementations (Required Fields Matrix + Policy Engine), we built a **unified Constraint Validator** that:

1. **Computes** what's missing based on category requirements
2. **Determines** what policy citations must appear
3. **Formats** constraints for LLM prompt injection
4. **Validates** responses post-generation
5. **Enforces** missing items by auto-appending them

### Key Insight

```
policy_service.py = "The Textbook"   → Full policy text for LLM understanding
constraint_validator.py = "The Exam Rubric" → Specific items that MUST appear
```

Both work together. The textbook provides context; the rubric enforces compliance.

---

## Architecture Overview

### Before (Disconnected)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Ticket    │────▶│  Extractor  │────▶│   Facts     │ (stored but not enforced)
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Policy    │────▶│   Service   │────▶│   Text      │ (context only)
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
                                        ┌─────────────┐
                                        │   Agent     │ (makes best effort)
                                        └─────────────┘
```

### After (Unified Enforcement)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Ticket    │────▶│  Extractor  │────▶│   Facts     │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌─────────────┐             │
                    │ Requirements│◀────────────┤
                    │   Matrix    │             │
                    └──────┬──────┘             │
                           │                   │
                           ▼                   ▼
                    ┌──────────────────────────────┐
                    │    CONSTRAINT VALIDATOR      │
                    │  ┌────────────────────────┐  │
                    │  │ missing = required -   │  │
                    │  │          present       │  │
                    │  └────────────────────────┘  │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴───────────────┐
                    │      Constraint Result       │
                    │  • missing_fields[]          │
                    │  • required_asks[]           │
                    │  • must_not_ask[]            │
                    │  • required_citations[]      │
                    └──────────────┬───────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Planner      │    │   React Agent   │    │ Draft Response  │
│ (injects into   │    │ (injects into   │    │ (post-validates │
│  exec plan)     │    │  agent context) │    │  & enforces)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## Components

### 1. Requirements Matrix (`app/config/requirements_matrix.py`)

**Purpose:** Define what fields are REQUIRED for each ticket category.

```python
REQUIREMENTS_MATRIX = {
    "warranty_claim": {
        "required": ["receipt", "address"],
        "conditional": {"photos": "if_defect_claim"},
        "policies": ["warranty_standard"],
        "product_specific_policies": {"hose": "hose_warranty"},
    },
    "missing_parts": {
        "required": ["po_number", "address"],
        "policies": ["missing_parts_window"],
    },
    # ... 13 categories total
}
```

**Key Features:**
- `required`: Fields that MUST be present
- `conditional`: Fields needed in specific contexts
- `policies`: Which policy rules apply to this category
- `product_specific_policies`: Product-dependent policy overrides

**Field Mappings:**
```python
FIELD_TO_FACTS_KEY = {
    "receipt": "receipt_provided",      # Maps to ticket_facts key
    "address": "shipping_address_provided",
    "po_number": "po_number",           # Checks if truthy
    "photos": "has_images",
}

FIELD_ASK_TEMPLATES = {
    "receipt": "Could you please provide your proof of purchase (receipt, invoice, or order confirmation)?",
    "address": "What shipping address should we send the replacement to?",
    # Customer-friendly ask messages
}
```

#### Complete Field Definitions (8 Fields)

| Field Key | Human-Readable Name | ticket_facts Key | Ask Template |
|-----------|---------------------|------------------|--------------|
| `receipt` | Proof of purchase (receipt, invoice, or order confirmation) | `has_receipt` | "Could you please provide your proof of purchase...?" |
| `address` | Shipping address for replacement delivery | `has_address` | "What address should we send the replacement to?" |
| `photos` | Photo(s) showing the issue or defect | `has_photos` | "Could you please send a photo showing the issue...?" |
| `video` | Video showing the issue | `has_video` | "If possible, could you send a short video...?" |
| `po` | PO number or order number | `has_po` | "Could you please provide your PO number...?" |
| `model` | Product model number | `has_model_number` | "Could you please provide the product model number...?" |
| `finish` | Product finish/color preference | `raw_finish_mentions` | "What finish/color would you prefer...?" |
| `part_number` | Specific part number needed | `raw_part_numbers` | "Could you please specify which part(s) you need?" |

#### Complete Requirements by Category (13 Categories)

| Category | Required Fields | Conditional Fields | Policies |
|----------|----------------|-------------------|----------|
| `warranty_claim` | receipt, address | photos, video | warranty_standard |
| `product_issue` | model | photos, receipt, address | warranty_standard |
| `missing_parts` | po, address | photos | missing_parts_window |
| `replacement_parts` | model, address | receipt, part_number, photos | warranty_standard |
| `return_refund` | receipt, address | photos | return_policy |
| `product_inquiry` | model | finish | (none) |
| `finish_color` | model | — | (none) |
| `installation_help` | model | photos | (none) |
| `pricing_request` | (none) | model, part_number | (none) |
| `dealer_inquiry` | (none) | — | dealer_program |
| `shipping_tracking` | po | — | (none) |
| `feedback_suggestion` | (none) | — | (none) |
| `general` | (none) | — | (none) |

### 2. Policy Rules (`app/config/policy_rules.py`)

**Purpose:** Define structured policy rules with specific citations that MUST appear in responses.

```python
POLICY_RULES = {
    "warranty_standard": {
        "policy_id": "warranty_standard",
        "name": "Standard Product Warranty",
        "coverage_months": 12,
        "citation": "Our standard warranty covers manufacturing defects for 12 months from the date of purchase.",
        "requires_receipt": True,
        "date_field": "purchase_date",
    },
    "hose_warranty": {
        "coverage_months": 24,  # Extended warranty for hoses
        "citation": "Our hose products are covered by an extended 2-year warranty against manufacturing defects.",
    },
    "return_policy": {
        "window_days": 45,
        "restocking_fee_percent": 15,
        "citation": "Items may be returned within 45 days of delivery. A 15% restocking fee applies to opened items.",
    },
}
```

#### Complete Policy Rules (6 Policies)

| Policy ID | Name | Key Parameters | Citation |
|-----------|------|----------------|----------|
| `warranty_standard` | Standard Product Warranty | 12 months coverage | "Our standard warranty covers manufacturing defects for 12 months from the date of purchase." |
| `hose_warranty` | Hose Extended Warranty | 24 months coverage | "Hoses and supply lines are covered under our extended 2-year warranty against manufacturing defects." |
| `lifetime_warranty` | Lifetime Warranty | Lifetime coverage | "This product includes our lifetime warranty against manufacturing defects." |
| `missing_parts_window` | Missing Parts Claim | 45 days window | "Missing parts must be reported within 45 days of delivery to be eligible for free replacement." |
| `return_policy` | Return Policy | 45 days, 15% restocking | "Returns are accepted within 45 days of purchase for unused items. A 15% restocking fee applies to opened items." |
| `dealer_program` | Dealer Program | Partnership inquiry | "We welcome dealer and distributor partnerships. Our team will review your application and follow up within 5-7 business days." |

**Helper Functions:**
```python
def check_warranty_coverage(purchase_date: str, coverage_months: int) -> dict:
    """Compute if warranty is still valid."""
    return {
        "is_covered": bool,
        "days_remaining": int,
        "expired_days_ago": int,
    }

def get_policies_for_product(product_text: str) -> List[str]:
    """Detect product-specific policies (e.g., 'hose' → hose_warranty)."""
```

### 3. Constraint Validator Service (`app/services/constraint_validator.py`)

**Purpose:** Main enforcement layer that computes and validates constraints.

#### Core Function: `validate_constraints()`

```python
def validate_constraints(
    ticket_facts: Dict[str, Any],
    ticket_category: str,
    product_text: Optional[str] = None,
) -> ConstraintResult:
    """
    Main entry point - validates ticket against requirements and policies.
    
    Returns ConstraintResult with:
        - missing_fields: ["receipt", "address"]
        - required_asks: ["Please provide your receipt...", ...]
        - present_fields: ["model", "email"]
        - must_not_ask: ["Model number (already provided: ABC123)", ...]
        - applicable_policies: ["warranty_standard", "hose_warranty"]
        - required_citations: ["Our warranty covers...", ...]
        - can_proceed: False  # Blocked until missing info received
        - blocking_missing: ["receipt"]  # Critical missing fields
    """
```

#### Formatting: `format_constraints_for_prompt()`

```python
def format_constraints_for_prompt(result: ConstraintResult) -> str:
    """
    Format constraints for LLM prompt injection.
    
    Output:
    ═══════════════════════════════════════════════════════════════════════
    🔒 MANDATORY CONSTRAINTS - YOU MUST FOLLOW THESE
    ═══════════════════════════════════════════════════════════════════════
    
    ❌ DO NOT ask for the following (already provided):
       • Model number (already provided: ABC-123)
       • Customer email (already provided: john@example.com)
    
    ✅ YOU MUST ask for the following (missing required info):
       • Could you please provide your proof of purchase?
       • What shipping address should we send the replacement to?
    
    📜 YOU MUST include these policy statements in your response:
       • "Our standard warranty covers manufacturing defects for 12 months..."
    
    ═══════════════════════════════════════════════════════════════════════
    """
```

#### Post-Validation: `post_validate_response()`

```python
def post_validate_response(
    response_text: str,
    constraints: ConstraintResult | Dict
) -> Dict[str, Any]:
    """
    Check if LLM response meets constraints.
    
    Returns:
        {
            "valid": False,
            "violations": ["Did not ask for required info: receipt"],
            "missing_citations": ["Our warranty covers..."],
            "unnecessary_asks": ["Asked for model which was already provided"],
            "warnings": [...],  # Combined list
        }
    """
```

#### Auto-Enforcement: `enforce_constraints_on_response()`

```python
def enforce_constraints_on_response(
    response_text: str,
    constraints: ConstraintResult | Dict
) -> str:
    """
    Auto-append missing citations and asks to response.
    
    If response is missing required elements, appends:
    
    **Policy Information:**
    • Our warranty covers manufacturing defects for 12 months...
    
    **To help us assist you better, please provide:**
    • Your proof of purchase (receipt or invoice)
    • Your shipping address
    """
```

### 4. State Extension (`app/graph/state.py`)

Added two new fields to `TicketState`:

```python
class TicketState(TypedDict, total=False):
    # ... existing fields ...
    
    # NEW: Constraint validation results
    constraint_result: Optional[Dict[str, Any]]
    constraints_prompt_section: Optional[str]
```

---

## Data Flow

### Step-by-Step Flow

```
1. TICKET ARRIVES
   │
   ▼
2. TICKET_EXTRACTOR
   │  Extracts: ticket_facts = {
   │      "receipt_provided": False,
   │      "shipping_address_provided": True,
   │      "po_number": "PO-12345",
   │      "model_number": "ABC-123",
   │      ...
   │  }
   │
   ▼
3. ROUTING_AGENT
   │  Classifies: ticket_category = "warranty_claim"
   │
   ▼
4. PLANNER (First Integration Point)
   │  Calls: validate_constraints(ticket_facts, "warranty_claim")
   │  
   │  Computes:
   │  • Required for warranty_claim: ["receipt", "address"]
   │  • Present: ["address"] (has shipping_address_provided=True)
   │  • Missing: ["receipt"] (receipt_provided=False)
   │  
   │  Generates:
   │  • required_asks: ["Please provide your proof of purchase..."]
   │  • must_not_ask: ["Shipping address (already provided)"]
   │  • required_citations: ["Our warranty covers 12 months..."]
   │  
   │  Stores in execution_plan:
   │  • _constraint_result: {...}
   │  • _constraints_prompt: "🔒 MANDATORY CONSTRAINTS..."
   │
   ▼
5. REACT_AGENT (Second Integration Point)
   │  Retrieves constraints from execution_plan
   │  Injects constraints_prompt into agent_context:
   │  
   │  ════════════════════════════════════════════════
   │  🔒 MANDATORY CONSTRAINTS - YOU MUST FOLLOW THESE
   │  ════════════════════════════════════════════════
   │  
   │  ❌ DO NOT ask for: Shipping address (already provided)
   │  ✅ YOU MUST ask for: Proof of purchase
   │  📜 YOU MUST cite: "Our warranty covers 12 months..."
   │  
   │  Agent sees this BEFORE making decisions
   │
   ▼
6. DRAFT_RESPONSE (Third Integration Point)
   │  
   │  A. PRE-GENERATION:
   │     Injects constraints_prompt into LLM prompt
   │  
   │  B. LLM GENERATES RESPONSE:
   │     "Thank you for your warranty claim. Our warranty
   │      covers products for 12 months. Could you please
   │      provide your proof of purchase?"
   │  
   │  C. POST-VALIDATION:
   │     Calls post_validate_response(response, constraints)
   │     Checks: ✓ Asked for receipt
   │             ✓ Cited warranty period
   │             ✓ Did NOT ask for address (already had it)
   │  
   │  D. AUTO-ENFORCEMENT (if needed):
   │     If validation failed, calls enforce_constraints_on_response()
   │     Appends missing citations/asks to response
   │
   ▼
7. FINAL RESPONSE
   Response is guaranteed to:
   • Ask for all missing required fields
   • Include all required policy citations
   • NOT ask for information already provided
```

---

## Technical Implementation

### ConstraintResult Dataclass

```python
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
    
    # Conditional info
    conditional_fields: Dict[str, str] = field(default_factory=dict)
    
    # Validation flags
    can_proceed: bool = True
    blocking_missing: List[str] = field(default_factory=list)
    
    # Metadata
    validation_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {...}
```

### Dual Input Support

All validator functions accept BOTH `ConstraintResult` dataclass AND `dict`:

```python
def format_constraints_for_prompt(result: ConstraintResult | Dict[str, Any]) -> str:
    # Handle both dataclass and dict input
    if isinstance(result, dict):
        must_not_ask = result.get("must_not_ask", [])
        required_asks = result.get("required_asks", [])
        # ...
    else:
        must_not_ask = result.must_not_ask
        required_asks = result.required_asks
        # ...
```

This allows:
1. Direct use after `validate_constraints()` returns dataclass
2. Use after retrieving from state (stored as dict via `.to_dict()`)

### Graceful Degradation

All integration points use try/except with availability flags:

```python
# At module level
try:
    from app.services.constraint_validator import validate_constraints, ...
    CONSTRAINT_VALIDATOR_AVAILABLE = True
except ImportError:
    CONSTRAINT_VALIDATOR_AVAILABLE = False
    logger.warning("Constraint validator not available")

# In function
if CONSTRAINT_VALIDATOR_AVAILABLE:
    try:
        constraint_result = validate_constraints(...)
    except Exception as e:
        logger.warning(f"Constraint validation failed: {e}")
        constraint_result = None
```

The system continues to work even if constraint validation fails.

---

## Integration Points

### 1. Planner Integration (`app/nodes/planner.py`)

```python
# Step 3.5: Constraint validation (after routing, before agent)
if CONSTRAINT_VALIDATOR_AVAILABLE:
    try:
        ticket_facts = state.get("ticket_facts", {})
        ticket_category = state.get("ticket_category", "general")
        
        constraint_result = validate_constraints(
            ticket_facts=ticket_facts,
            ticket_category=ticket_category,
            product_text=state.get("product_text", "")
        )
        
        response["_constraint_result"] = constraint_result.to_dict()
        response["_constraints_prompt"] = format_constraints_for_prompt(constraint_result)
```

### 2. React Agent Integration (`app/nodes/react_agent.py`)

```python
# Get constraints from execution plan
if execution_plan.get("_constraint_result"):
    constraint_result = execution_plan.get("_constraint_result")
    constraints_prompt = execution_plan.get("_constraints_prompt", "")

# Inject into agent context (appears before agent reasoning)
if constraints_prompt:
    constraint_injection_point = agent_context.find("═══ PREVIOUS ITERATIONS")
    if constraint_injection_point > 0:
        agent_context = (
            agent_context[:constraint_injection_point] +
            constraints_prompt + "\n\n" +
            agent_context[constraint_injection_point:]
        )
```

### 3. Draft Response Integration (`app/nodes/response/draft_response.py`)

```python
# PRE-GENERATION: Add to prompt
user_prompt = f"""
CUSTOMER TICKET: {ticket_text}
TICKET CATEGORY: {ticket_category}
{policy_prompt_section}
{constraints_prompt_section}  # ← NEW
RETRIEVED CONTEXT: {context}
"""

# POST-VALIDATION: Check response
if CONSTRAINT_VALIDATOR_AVAILABLE and constraint_result:
    validation = post_validate_response(
        response_text=response_text,
        constraints=constraint_result
    )
    
    if not validation.get("valid"):
        # Auto-enforce missing items
        response_text = enforce_constraints_on_response(
            response_text=response_text,
            constraints=constraint_result
        )
```

---

## Configuration Guide

### Adding a New Ticket Category

1. **Add to REQUIREMENTS_MATRIX** (`app/config/requirements_matrix.py`):

```python
REQUIREMENTS_MATRIX["new_category"] = {
    "required": ["field1", "field2"],
    "conditional": {"field3": "if_condition"},
    "policies": ["relevant_policy"],
}
```

2. **Add field mappings** (if new fields):

```python
FIELD_TO_FACTS_KEY["field1"] = "ticket_facts_key_name"
FIELD_NAMES["field1"] = "Human Readable Name"
FIELD_ASK_TEMPLATES["field1"] = "Customer-friendly ask message"
```

3. **Add aliases** (optional):

```python
CATEGORY_ALIASES["alternate_name"] = "new_category"
```

### Adding a New Policy Rule

Add to POLICY_RULES (`app/config/policy_rules.py`):

```python
POLICY_RULES["new_policy"] = {
    "policy_id": "new_policy",
    "name": "New Policy Name",
    "citation": "The exact text that MUST appear in responses.",
    # Optional fields based on policy type:
    "coverage_months": 12,
    "window_days": 45,
    "requires_receipt": True,
}
```

### Adding Product-Specific Policies

Add to POLICY_TRIGGERS:

```python
POLICY_TRIGGERS["keyword_in_product"] = "policy_id"
# e.g., POLICY_TRIGGERS["hose"] = "hose_warranty"
```

---

## Testing

### Unit Test: Constraint Validation

```python
from app.services.constraint_validator import validate_constraints

# Test warranty claim with missing receipt
result = validate_constraints(
    ticket_facts={
        "receipt_provided": False,
        "shipping_address_provided": True,
    },
    ticket_category="warranty_claim",
)

assert "receipt" in result.missing_fields
assert "address" not in result.missing_fields  # Already provided
assert result.can_proceed == False  # Blocked by missing receipt
assert len(result.required_citations) > 0  # Has warranty citation
```

### Unit Test: Post-Validation

```python
from app.services.constraint_validator import post_validate_response

validation = post_validate_response(
    response_text="Thank you! We'll process your warranty claim.",
    constraints=result.to_dict()
)

assert validation["valid"] == False  # Missing receipt ask, missing citation
assert "receipt" in str(validation["violations"])
```

### Integration Test

```python
# Run with: python test_workflow_manual.py

# Create test ticket
ticket = {
    "subject": "Warranty claim for broken hose",
    "description": "My hose is leaking. I bought it 3 months ago.",
    # Note: No receipt, no address provided
}

# Run workflow
result = run_workflow(ticket)

# Verify constraints enforced
assert "proof of purchase" in result["draft_response"].lower()
assert "shipping address" in result["draft_response"].lower()
assert "warranty" in result["draft_response"].lower()
assert "12 months" in result["draft_response"].lower() or "2 year" in result["draft_response"].lower()
```

---

## Summary: Gap Resolution

| Gap | Solution | Implementation |
|-----|----------|----------------|
| No field requirements matrix | Created `REQUIREMENTS_MATRIX` | `app/config/requirements_matrix.py` |
| Extraction without enforcement | Unified validator computes `missing = required - present` | `validate_constraints()` |
| Policy TEXT vs RULES | Separated citations from full text | `POLICY_RULES` with `citation` field |
| No post-validation | Added response checking | `post_validate_response()` |
| No "must not ask" logic | Track present fields | `must_not_ask` in ConstraintResult |
| Inconsistent enforcement | Three-point injection (planner→agent→response) | All nodes receive same constraints |

---

## Files Created/Modified

### New Files

| File | Purpose |
|------|---------|
| `app/config/requirements_matrix.py` | Required fields per category |
| `app/config/policy_rules.py` | Policy rules with citations |
| `app/services/constraint_validator.py` | Main validator service |

### Modified Files

| File | Changes |
|------|---------|
| `app/graph/state.py` | Added `constraint_result`, `constraints_prompt_section` |
| `app/nodes/planner.py` | Added constraint validation call |
| `app/nodes/react_agent.py` | Added constraint injection |
| `app/nodes/response/draft_response.py` | Added pre/post validation |

---

## Future Enhancements

1. **Admin UI for Rules**: Web interface to edit requirements matrix without code changes
2. **A/B Testing**: Compare response quality with/without constraint enforcement
3. **Analytics Dashboard**: Track which constraints are most often violated
4. **Dynamic Policies**: Load policy rules from database for real-time updates
5. **Severity Levels**: Distinguish between blocking vs warning constraints

---

*Last Updated: February 2026*
*Implementation Version: 1.0*
