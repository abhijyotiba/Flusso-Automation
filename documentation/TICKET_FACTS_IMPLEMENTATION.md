# Ticket Facts Implementation Summary

## 📋 Overview

This document details the implementation of **Suggestion #1: Ticket Intake Extractor** - a system that creates a structured `ticket_facts` record to capture what information is present in a customer ticket before the ReACT agent begins processing.

**Implementation Date:** January 29, 2026

---

## 🎯 Problem Statement

### The Issue
The existing workflow had a critical flaw: the ReACT agent would frequently ask customers for information that was **already provided** in their ticket. 

**Example from Ticket #97841:**
- Customer clearly mentioned model numbers: `PBV1005` and `PBV2105`
- The system set confidence to only 20% and generated a response asking: *"Could you please provide the model number?"*
- This happened because the evidence resolver only checked tool outputs, not the original ticket content

### Root Cause Analysis
1. **No structured record** of what the customer provided upfront
2. **Model numbers were buried** in ticket text and not extracted systematically
3. **Planner had no visibility** into what information was already available
4. **Evidence resolver** couldn't differentiate between "missing info" vs "info not yet looked up"

---

## 🏗️ Solution Architecture

### Three-Tier Extraction System

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TICKET_FACTS RECORD                         │
├─────────────────────────────────────────────────────────────────────┤
│  TIER 1: DETERMINISTIC (set by ticket_extractor)                   │
│  ├── has_address, has_receipt, has_po, has_photos, has_video       │
│  ├── raw_product_codes: [{model, finish_code, finish_name}, ...]   │
│  ├── raw_part_numbers, raw_finish_mentions                         │
│  └── extracted_address, customer_name, requester_email             │
├─────────────────────────────────────────────────────────────────────┤
│  TIER 2: LLM-VERIFIED (set by planner)                             │
│  ├── planner_verified: bool                                        │
│  ├── planner_verified_models: ["PBV1005", "TRM.TVH.0211"]         │
│  ├── planner_verified_finishes: ["BB", "CP"]                      │
│  └── planner_corrections: {original: corrected}                    │
├─────────────────────────────────────────────────────────────────────┤
│  TIER 3: TOOL-CONFIRMED (set by react_agent)                       │
│  ├── confirmed_model: str                                          │
│  ├── confirmed_model_source: "catalog" | "ocr" | "vision"         │
│  ├── confirmed_model_confidence: float                             │
│  └── confirmed_finish, confirmed_parts                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

| Principle | Description |
|-----------|-------------|
| **Not Compulsory** | Fields describe what IS present, not what's required |
| **Fully Mutable** | Any downstream node can update any field |
| **Hints, Not Constraints** | Used to reduce iterations, not block processing |
| **Audit Trail** | Every update is tracked with timestamp and source |

---

## 📁 Files Modified

| File | Change Type | Key Changes |
|------|-------------|-------------|
| `app/nodes/ticket_extractor.py` | **NEW** | Core extraction node with product code parsing |
| `app/graph/state.py` | Modified | Added `ticket_facts` field to state |
| `app/graph/graph_builder_react.py` | Modified | Added ticket_extractor node to flow |
| `app/nodes/planner.py` | Modified | Added `verify_ticket_facts()` function |
| `app/nodes/react_agent.py` | Modified | Integrated ticket_facts, passes to evidence resolver |
| `app/nodes/react_agent_helpers.py` | Modified | Updated `_build_agent_context()` with hints |
| `app/nodes/evidence_resolver.py` | Modified | **Critical fix:** Added ticket_facts checking |

---

### 1. NEW: `app/nodes/ticket_extractor.py`

**Purpose:** Core extraction node that runs immediately after `fetch_ticket`

**Key Components:**

```python
# Finish Code Dictionary
FINISH_CODES = {
    "CP": "Chrome",
    "BN": "Brushed Nickel PVD",
    "PN": "Polished Nickel PVD",
    "MB": "Matte Black",
    "SB": "Satin Brass PVD",
    "BB": "Brushed Bronze PVD",
    # ... more codes
}
```

**Functions:**

| Function | Purpose |
|----------|---------|
| `parse_product_code(full_code)` | Splits `TRM.TVH.0211BB` → model: `TRM.TVH.0211`, finish: `BB` |
| `extract_product_codes(text)` | Regex extraction of all product codes from text |
| `extract_ticket_facts(state)` | Main node function - creates the ticket_facts record |
| `update_ticket_facts(facts, updates, updated_by)` | Helper for mutable updates with audit trail |
| `get_model_candidates_from_facts(facts)` | Returns models in priority order (confirmed → verified → raw) |

**Why This Was Needed:**
- Deterministic extraction is faster and more reliable than LLM for pattern matching
- Product code format (model + finish suffix) is domain-specific knowledge
- Provides a single source of truth for "what did the customer tell us"

---

### 2. MODIFIED: `app/graph/state.py`

**Change:** Added `ticket_facts: Optional[Dict[str, Any]]` field to `TicketState`

**Location:** Lines 76-119 (new section after `product_match_reasoning`)

```python
# ==========================================
# TICKET FACTS (from ticket_extractor node)
# Mutable record of what's known about the ticket
# ==========================================
ticket_facts: Optional[Dict[str, Any]]
```

**Why This Was Needed:**
- LangGraph requires all state fields to be declared in the TypedDict
- Provides type hints for IDE support
- Documents the expected structure for future maintainers

---

### 3. MODIFIED: `app/graph/graph_builder_react.py`

**Changes:**

1. **Added import:**
```python
from app.nodes.ticket_extractor import extract_ticket_facts
```

2. **Added node:**
```python
graph.add_node("ticket_extractor", extract_ticket_facts)
```

3. **Updated flow:**
```python
# OLD: fetch_ticket → routing
# NEW: fetch_ticket → ticket_extractor → routing
graph.add_edge("fetch_ticket", "ticket_extractor")
graph.add_edge("ticket_extractor", "routing")
```

**New Graph Flow:**
```
fetch_ticket → ticket_extractor → routing → [skip_handler OR react_agent] → ...
```

**Why This Was Needed:**
- Extraction must happen BEFORE routing/planning so facts are available
- Must be early in the pipeline to benefit all downstream nodes

---

### 4. MODIFIED: `app/nodes/planner.py`

**Changes:**

1. **New import:**
```python
from app.nodes.ticket_extractor import update_ticket_facts, get_model_candidates_from_facts
```

2. **New function: `verify_ticket_facts(state)`**
   - Uses LLM to verify raw regex extractions
   - Corrects parsing errors (e.g., `100.1170C` → `100.1170CP`)
   - Identifies missed model numbers
   - Updates `planner_verified_models` and `planner_verified_finishes`

3. **New helper: `_format_ticket_facts_for_planner(state)`**
   - Formats ticket_facts into human-readable summary for the planning prompt

4. **Updated `PLANNING_PROMPT`:**
   - Added new section: `🔍 PRE-EXTRACTED TICKET FACTS`
   - Instructs planner to use facts to optimize tool selection
   - Example: "If product codes are already extracted, skip blind vision search"

**Why This Was Needed:**
- Regex can have false positives (order numbers misidentified as models)
- LLM verification adds semantic understanding
- Planner needs visibility into facts to make smarter tool choices

---

### 5. MODIFIED: `app/nodes/react_agent.py`

**Changes:**

1. **New imports:**
```python
from app.nodes.planner import verify_ticket_facts
from app.nodes.ticket_extractor import update_ticket_facts, get_model_candidates_from_facts
```

2. **New variables in planning phase:**
```python
ticket_facts = state.get("ticket_facts", {}) or {}
ticket_facts_updates = {}
```

3. **Verification call added:**
```python
if ticket_facts and not ticket_facts.get("planner_verified"):
    verified_result = verify_ticket_facts(state)
    if verified_result.get("ticket_facts"):
        ticket_facts = verified_result["ticket_facts"]
        ticket_facts_updates = {"ticket_facts": ticket_facts}
```

4. **Context building updated:**
```python
agent_context = _build_agent_context(
    ...,
    ticket_facts=ticket_facts  # NEW parameter
)
```

5. **Return updated to include ticket_facts:**
```python
if ticket_facts_updates:
    result.update(ticket_facts_updates)
```

6. **Error path also preserves ticket_facts:**
```python
**(ticket_facts_updates if ticket_facts_updates else {}),
```

**Why This Was Needed:**
- React agent is where planning happens, so verification fits here
- Facts need to flow into agent context for tool guidance
- Both success and error paths should preserve any verified facts

---

### 6. MODIFIED: `app/nodes/react_agent_helpers.py`

**Changes:**

1. **Updated `_build_agent_context()` signature:**
```python
def _build_agent_context(
    ...,
    ticket_facts: Optional[Dict[str, Any]] = None  # NEW parameter
) -> str:
```

2. **New context section for ticket_facts:**
```python
if ticket_facts:
    context_parts.append(f"\n\n═══ PRE-EXTRACTED HINTS ═══")
    
    # Show model candidates prominently
    if model_candidates:
        context_parts.append(f"\n🎯 MODEL CANDIDATES:")
        context_parts.append(f"   ➡️ Use product_catalog_tool with these BEFORE vision search!")
    
    # Show finish preferences
    # Show presence hints (receipt, PO, address)
```

**Why This Was Needed:**
- Agent needs to SEE the hints to use them
- Model candidates are shown prominently with instruction to search first
- Reduces wasted iterations on blind vision/OCR when models are known

---

### 7. MODIFIED: `app/nodes/__init__.py`

**Changes:**
```python
from app.nodes.ticket_extractor import (
    extract_ticket_facts,
    update_ticket_facts,
    get_model_candidates_from_facts,
    parse_product_code,
    FINISH_CODES
)

__all__ = [
    # ... existing exports
    "extract_ticket_facts",
    "update_ticket_facts",
    "get_model_candidates_from_facts",
    "parse_product_code",
    "FINISH_CODES"
]
```

**Why This Was Needed:**
- Clean public API for importing ticket_extractor functions
- Allows other modules to use helpers without deep imports

---

## 🔄 Data Flow

```
┌──────────────┐
│ fetch_ticket │ Fetches ticket from Freshdesk
└──────┬───────┘
       │ ticket_text, ticket_subject, ticket_attachments
       ▼
┌──────────────────┐
│ ticket_extractor │ Creates ticket_facts (TIER 1: Deterministic)
└──────┬───────────┘
       │ ticket_facts: {raw_product_codes, has_receipt, has_address, ...}
       ▼
┌──────────┐
│ routing  │ Classifies ticket category
└──────┬───┘
       │
       ▼
┌─────────────┐
│ react_agent │ 
│  ├─ verify_ticket_facts() → Updates TIER 2 (LLM-verified)
│  ├─ create_execution_plan() → Uses facts for smarter planning
│  └─ tool loop → Can update TIER 3 (confirmed_model)
└─────────────┘
```

---

## 📊 Expected Impact

### Before Implementation
- ❌ Model numbers in ticket text were ignored
- ❌ Agent asked for info already provided
- ❌ Many wasted iterations on redundant searches
- ❌ Evidence resolver set low confidence for valid tickets

### After Implementation
- ✅ Model numbers extracted and shown as hints
- ✅ Agent sees "Use product_catalog_tool with PBV1005 BEFORE vision search"
- ✅ Fewer iterations needed (model candidates are prioritized)
- ✅ Evidence resolver can check if info was in original ticket

---

## 🧪 Testing Verification

All tests passed:

| Test | Result |
|------|--------|
| Module imports | ✅ Pass |
| Product code parsing (`TRM.TVH.0211BB` → model + finish) | ✅ Pass |
| Ticket extraction (product codes, addresses, flags) | ✅ Pass |
| Graph building with new node | ✅ Pass |
| Agent context includes ticket_facts hints | ✅ Pass |
| Edge cases (None, empty dict) | ✅ Pass |
| IDE syntax check (Pylance) | ✅ No errors |

---

## 📝 Example Output

For ticket text:
> "I need replacement parts for my faucet TRM.TVH.0211BB and PBV2105. My address is 123 Main St, Austin TX 78701."

**Extracted ticket_facts:**
```json
{
  "has_address": true,
  "has_receipt": false,
  "has_photos": false,
  "raw_product_codes": [
    {"full_sku": "TRM.TVH.0211BB", "model": "TRM.TVH.0211", "finish_code": "BB", "finish_name": "Brushed Bronze PVD"},
    {"full_sku": "PBV2105", "model": "PBV2105", "finish_code": null, "finish_name": null}
  ],
  "extracted_address": "123 Main St, Austin TX 78701",
  "planner_verified": false
}
```

**Agent context shows:**
```
═══ PRE-EXTRACTED HINTS (use to guide your search) ═══

🎯 MODEL CANDIDATES (try these in product_catalog_tool):
   📦 TRM.TVH.0211 (finish: BB)
   📦 PBV2105
   ➡️ Use product_catalog_tool with these BEFORE doing vision search!

⚡ HINTS:
   📍 Shipping address may be present
```

---

### 7. MODIFIED: `app/nodes/evidence_resolver.py` (Critical Gap Fix)

**Problem Identified:** The original implementation didn't connect ticket_facts to evidence_resolver, creating a "Split Brain" situation where the agent knew about models but evidence_resolver could still trigger "request_info".

**Changes:**

1. **Updated `analyze_evidence()` signature:**
```python
def analyze_evidence(
    ocr_result: Optional[Dict[str, Any]] = None,
    vision_result: Optional[Dict[str, Any]] = None,
    product_search_results: Optional[List[Dict[str, Any]]] = None,
    document_results: Optional[List[Dict[str, Any]]] = None,
    past_ticket_results: Optional[List[Dict[str, Any]]] = None,
    catalog_lookup_func = None,
    agent_identified_product: Optional[Dict[str, Any]] = None,
    agent_confidence: float = 0.0,
    ticket_facts: Optional[Dict[str, Any]] = None  # NEW: Pre-extracted ticket facts
) -> EvidenceBundle:
```

2. **Added ticket_facts evidence processing (after OCR, priority 1.5):**
```python
# 1.5. NEW: Process ticket_facts (pre-extracted from ticket text)
if ticket_facts:
    raw_codes = ticket_facts.get("raw_product_codes", [])
    verified_models = ticket_facts.get("verified_models", [])
    models_to_use = verified_models if verified_models else raw_codes
    
    for model_code in models_to_use:
        item = EvidenceItem(
            source="ticket_facts",
            product_model=model_code,
            confidence=0.85 if verified_models else 0.75,
            is_exact_match=True  # Customer explicitly mentioned this
        )
        evidence_items.append(item)
```

3. **Added PRIORITY 2.5 in conflict resolution:**
```python
# PRIORITY 2.5: Ticket Facts models (customer-provided in ticket text)
# This prevents asking for model numbers the customer already gave us
ticket_facts_items = [i for i in evidence_items if i.source == "ticket_facts"]
if ticket_facts_items:
    best_ticket_fact = max(ticket_facts_items, key=lambda x: x.confidence)
    bundle.primary_product = {...}
    bundle.resolution_action = "proceed"  # NOT "request_info"!
    return bundle
```

4. **Updated `generate_info_request_response()` signature:**
```python
def generate_info_request_response(
    bundle: EvidenceBundle,
    customer_name: str = "Customer",
    ticket_subject: str = "",
    ticket_text: str = "",
    ticket_category: str = "",
    ticket_facts: Optional[Dict[str, Any]] = None  # NEW
) -> Dict[str, str]:
```

5. **Enhanced private notes with customer-provided info status:**
```python
private_note = f"""🤖 **AI Summary**
• Request: {request_type}
• Confidence: {int(bundle.final_confidence * 100)}%

**Customer-Provided Info:**
✅ Model(s) provided: PBV1005, PBV2105
✅ Receipt: Present
"""
```

**Why This Was Critical:**
- Without this fix, the evidence_resolver operated blind to ticket_facts
- Could still set `resolution_action = "request_info"` even when models were known
- This was the final piece needed to fully solve the original problem (Ticket #97841)

---

## 🔮 Future Enhancements

1. **React agent updating confirmed_model** - When product_catalog_tool confirms a model, update `ticket_facts.confirmed_model`
2. ~~**Evidence resolver integration**~~ ✅ COMPLETED - Check ticket_facts when deciding if info is "missing"
3. **Address parsing improvement** - Better regex for international addresses
4. **Finish preference extraction** - Parse "I want it in matte black" → `raw_finish_mentions`

---

## 📚 Related Files

- [app/nodes/ticket_extractor.py](../app/nodes/ticket_extractor.py) - Core extraction logic
- [app/graph/state.py](../app/graph/state.py) - State schema with ticket_facts
- [app/graph/graph_builder_react.py](../app/graph/graph_builder_react.py) - Graph flow
- [app/nodes/planner.py](../app/nodes/planner.py) - Verification and planning
- [app/nodes/react_agent.py](../app/nodes/react_agent.py) - Integration point
- [app/nodes/react_agent_helpers.py](../app/nodes/react_agent_helpers.py) - Context building
- [app/nodes/evidence_resolver.py](../app/nodes/evidence_resolver.py) - Evidence analysis with ticket_facts
