"""
ReACT Agent Node - FIXED VERSION
Reasoning + Acting Loop with proper context tracking
Enhanced with Planning Module (Phase 1)
Enhanced with Ticket Facts (Phase 2)
Enhanced with Constraint Validation (Phase 3)
"""

import logging
import time
import json
from typing import Dict, Any, List

from app.graph.state import TicketState, ReACTIteration
from app.clients.llm_client import get_llm_client
from app.utils.audit import add_audit_event
from app.config.settings import settings

from app.nodes.react_agent_helpers import (
    _build_agent_context,
    _execute_tool,
    _populate_legacy_fields,

)

# Planning module import
try:
    from app.nodes.planner import (
        create_execution_plan, 
        get_plan_context_for_agent,
        should_follow_plan_step,
        verify_ticket_facts  # NEW: For ticket_facts verification
    )
    PLANNER_AVAILABLE = True
except ImportError:
    PLANNER_AVAILABLE = False
    logging.getLogger(__name__).warning("Planner module not available")

# Ticket extractor helpers import
try:
    from app.nodes.ticket_extractor import (
        update_ticket_facts,
        get_model_candidates_from_facts
    )
    TICKET_EXTRACTOR_AVAILABLE = True
except ImportError:
    TICKET_EXTRACTOR_AVAILABLE = False
    logging.getLogger(__name__).warning("Ticket extractor module not available")

# Evidence resolver import
try:
    from app.nodes.evidence_resolver import (
        analyze_evidence,
        EvidenceItem,
        generate_info_request_response,
        should_request_more_info,
        VISION_HIGH_THRESHOLD,
        VISION_MEDIUM_THRESHOLD
    )
    EVIDENCE_RESOLVER_AVAILABLE = True
except ImportError:
    EVIDENCE_RESOLVER_AVAILABLE = False
    logging.getLogger(__name__).warning("Evidence resolver module not available")

# Constraint validator import (NEW)
try:
    from app.services.constraint_validator import (
        validate_constraints,
        format_constraints_for_prompt,
        format_constraints_summary,
    )
    CONSTRAINT_VALIDATOR_AVAILABLE = True
except ImportError:
    CONSTRAINT_VALIDATOR_AVAILABLE = False
    logging.getLogger(__name__).warning("Constraint validator not available")

logger = logging.getLogger(__name__)
STEP_NAME = "🤖 REACT_AGENT"

MAX_ITERATIONS = 15

# COMPREHENSIVE System Prompt with All Tools
REACT_SYSTEM_PROMPT = """You are an intelligent support agent helping resolve customer tickets for Flusso Kitchen & Bath company.

Your goal: Gather ALL necessary information to help the customer by using available tools strategically and efficiently.

═══════════════════════════════════════════════════════════════════════
⚠️ CRITICAL RULE: ANALYZE ALL ATTACHMENTS FIRST
═══════════════════════════════════════════════════════════════════════

🔴 MANDATORY FIRST STEP: If the ticket has ANY attachments (images, PDFs, documents):
   1. Use attachment_analyzer_tool or ocr_image_analyzer_tool to analyze ALL attachments
   2. Extract ALL available information (model numbers, part numbers, descriptions, order numbers from shipping labels)
   3. THEN decide which tools to use based on what you found
   4. DO NOT decide on tools before analyzing attachments

This prevents:
- Using wrong tools for extracted information (e.g., searching order numbers in product docs)
- Missing critical information in attachments
- Inefficient tool selection

═══════════════════════════════════════════════════════════════════════
📚 KNOWLEDGE SOURCES - WHAT INFORMATION IS AVAILABLE WHERE
═══════════════════════════════════════════════════════════════════════

1. **Gemini File Search (document_search_tool)** 📖 ⭐ PRIMARY KNOWLEDGE BASE
   ✅ HAS: Product specification PDFs with technical diagrams
   ✅ HAS: Parts diagrams showing product components and specific parts
   ✅ HAS: Parts lists with part numbers and MSRP/pricing
   ✅ HAS: Installation guides and manuals (step-by-step instructions)
   ✅ HAS: Core policy documents:
      - Warranty policy
      - MAP (Minimum Advertised Price) agreement
      - Training manuals
      - Return/refund policies
   ✅ HAS: Troubleshooting guides and FAQs
   ✅ HAS: Detailed product dimensions, materials, features
   ❌ DOES NOT HAVE: Customer order data, purchase orders, invoices, shipping/tracking info
   ❌ DOES NOT HAVE: Customer account information or order history
   
   🔴 USE THIS TOOL EXTENSIVELY - It contains the most detailed technical information!
   🔴 For ANY product-specific question, search with model number + specific topic
   🔴 For installation, search "[model] installation" or "[model] install guide"
   🔴 For parts/components, search "[model] parts" or "[model] parts diagram"
   
   ⚠️ CRITICAL: Order numbers from shipping labels are NOT product identifiers. 
   Do not search order numbers in this tool - it will return irrelevant results.

2. **Product Catalog (product_catalog_tool)** ⭐ COMPREHENSIVE DATABASE
   - 5,687 products with FULL details (70 fields per product)
   - Model numbers, group numbers, UPC codes
   - ALL finish variations (29 finishes: Chrome, Matte Black, Brushed Nickel, etc.)
   - Pricing (List Price, MAP Price)
   - Specifications (dimensions, weight, flow rate, holes needed)
   - Features and bullet points
   - Direct links to spec sheets, install manuals, parts diagrams
   - Video links (installation, operational, lifestyle)
   - Categories: Showering, Bathing, Sink Faucets, Kitchen, Bath Accessories, Spare Parts
   - Collections: Serie 100, Serie 196, Universal Fixtures, Cascade, etc.

3. **Past Tickets (past_tickets_search_tool)**
   - Similar resolved issues
   - Common solutions and resolutions
   - Customer communication patterns

4. **Vision/OCR Tools**
   - Product identification from images
   - Text extraction from labels, invoices, documents

5. **Spare Parts Pricing (spare_parts_pricing_tool)** ← NEW
   - Pricing for ~950 spare parts, components, replacement parts
   - Parts NOT in main product catalog
   - Handles, cartridges, valves, trims, accessories

═══════════════════════════════════════════════════════════════════════
🎯 TICKET TYPE HANDLING - DIFFERENT APPROACHES FOR DIFFERENT QUERIES
═══════════════════════════════════════════════════════════════════════

**PRODUCT ISSUES** (defects, broken, warranty, replacement parts):
  → Use vision/OCR to identify product
  → Verify with product_search
  → Find relevant docs and past tickets

**SPARE PARTS PRICING** (price for replacement handles, cartridges, trims, components):
  → Use spare_parts_pricing_tool with the part number
  → This tool has ~950 spare parts NOT in main product catalog
  → Common part prefixes: TVL, TVH, TRM, RP, PBV, MEM, K.
  → Example: "How much is TVH.5007?" → spare_parts_pricing_tool({{"part_number": "TVH.5007"}})

**FULL PRODUCT PRICING** (price for complete faucets, shower systems, etc.):
  → Use product_catalog_tool (has 5,687 complete products with prices)
  → NOT spare_parts_pricing_tool

**DEALER/PARTNERSHIP INQUIRIES** (becoming a dealer, open account):
  → Use document_search_tool to find dealer program information
  → Search for "dealer application", "partnership", "requirements"
  → NO need for product identification
  
**INSTALLATION HELP**:
  → Identify product first
  → Use document_search for installation guides
  
**GENERAL INQUIRIES**:
  → Use document_search with the query directly
  → NO forced product identification

═══════════════════════════════════════════════════════════════════════
📋 COMPLETE TOOL REGISTRY (You MUST use these tool names exactly)
═══════════════════════════════════════════════════════════════════════

1. **attachment_analyzer_tool** [PRIORITY: 1⭐]
   - PURPOSE: Extract structured data from PDFs, invoices, warranty docs, spec sheets
   - USE FIRST if customer has uploaded attachments!
   - OUTPUT: Model numbers, serial numbers, specifications, extracted text
   - PARAM: action_input = {{"attachments": [...], "focus": "model_numbers"}}

2. **attachment_type_classifier_tool** [PRIORITY: 2]
   - PURPOSE: Categorize attachment types (invoice, manual, spec sheet, warranty, etc.)
   - USE AFTER: attachment_analyzer to understand doc types
   - OUTPUT: Document type classification, confidence scores
   - PARAM: action_input = {{"attachments": [...]}}

3. **multimodal_document_analyzer_tool** [PRIORITY: 3]
   - PURPOSE: Intelligent document analysis with classification and data extraction
   - USE FOR: PDFs, documents with diagrams, contracts, agreements, licenses
   - DOCUMENT TYPES: invoice, spec_sheet, warranty_document, installation_manual,
                     shipping_document, return_authorization, agreement,
                     dealership_application, license, correspondence, catalog
   - OUTPUT: Document type, confidence, extracted data, identifiers (model/part/order numbers)
   - PARAM: action_input = {{"attachments": [...]}}

4. **ocr_image_analyzer_tool** [PRIORITY: 4]
   - PURPOSE: Intelligent image analysis with classification and text extraction
   - USE FOR: Customer photos of products, labels, receipts, packaging, damage
   - IMAGE TYPES: product_label, packaging, purchase_receipt, installed_product,
                  damaged_product, error_display, model_plate, general
   - OUTPUT: Image type, confidence, extracted model numbers, visible text
   - PARAM: action_input = {{"image_urls": [...]}}

5. **product_catalog_tool** [PRIORITY: 5⭐⭐] ← COMPREHENSIVE PRODUCT DATABASE
   - PURPOSE: Search 5,687 products with FULL details from the Flusso catalog
   - USE WHEN: Need product info, verify model numbers, find pricing, check availability
   
   SEARCH CAPABILITIES:
   • Exact model search: model_number="100.1170CP" → exact product
   • Group search: model_number="100.1170" → ALL finish variations (CP, BN, MB, SB...)
   • Fuzzy search: model_number="100.117" → suggests "100.1170" (typo correction)
   • Keyword search: query="floor mount tub faucet chrome"
   • Category filter: category="Bathing" or "Kitchen" or "Showering"
   • Collection filter: collection="Serie 100" or "Universal Fixtures"
   
   DATA RETURNED:
   • Product: model_no, title, category, sub_category, collection
   • Finish: finish_code (CP, BN, MB), finish_name (Chrome, Brushed Nickel)
   • Pricing: list_price, map_price
   • Specs: dimensions, weight, flow_rate_gpm, holes_needed
   • Resources: spec_sheet_url, install_manual_url, parts_diagram_url, install_video_url
   • Features: bullet points describing product
   • Variations: all available finishes for a product group
   • Related: spare parts for the product
   
   FINISH CODES: CP=Chrome, BN=Brushed Nickel, PN=Polished Nickel, MB=Matte Black,
                 SB=Satin Brass, BB=Brushed Bronze, GW=Gloss White, SS=Stainless Steel
   
   PARAM: action_input = {{"model_number": "100.1170"}} OR {{"query": "floor mount faucet"}}
          Optional: category, collection, include_variations, limit

6. **vision_search_tool** [PRIORITY: 6]
   - PURPOSE: Identify products from customer images using vision AI
   - USE WHEN: No model number found but customer sent images
   - OUTPUT: Identified product, match quality, confidence score
   - PARAM: action_input = {{"image_urls": [...]}}

7. **document_search_tool** [PRIORITY: 3⭐⭐⭐] ← PRIMARY DOCUMENTATION SOURCE
   - PURPOSE: Search ALL product documentation, specifications, parts lists, installation guides, policies
   - THIS IS YOUR MAIN KNOWLEDGE BASE - Use extensively!
   
   CONTAINS:
   • Product specification PDFs (dimensions, materials, features)
   • Parts diagrams with part numbers and MSRP pricing
   • Installation manuals with step-by-step instructions
   • Troubleshooting guides and FAQs
   • Policy documents (warranty, returns, MAP agreement)
   
   WHEN TO USE:
   • ALWAYS after identifying a product - get detailed specs and parts info
   • For installation questions - search "[model] installation guide"
   • For parts/pricing - search "[model] parts" or "[part_number] price"
   • For troubleshooting - search "[model] troubleshooting" or symptom description
   • For policies - search "warranty policy", "return policy", "dealer program"
   
   BEST PRACTICES:
   • Be specific in queries - include model numbers when known
   • Search multiple times with different queries if first result is insufficient
   • Combine with product_context parameter for better results
   
   - PARAM: action_input = {{"query": "search term", "product_context": "model_number or product_name"}}

8. **past_tickets_search_tool** [PRIORITY: 8]
   - PURPOSE: Find similar resolved tickets and solutions
   - USE NEAR END: After identifying product, before finishing
   - OUTPUT: Similar tickets, resolutions, common solutions, patterns
   - PARAM: action_input = {{"query": "search term"}}

9. **spare_parts_pricing_tool** [PRIORITY: 7] ← SPARE PARTS / COMPONENTS PRICING
   - PURPOSE: Look up pricing for SPARE PARTS, replacement components, accessories, extensions
   - ~950 spare parts with pricing that are NOT in the main product catalog
   
   ═══════════════════════════════════════════════════════════════════════
   ⚠️ CRITICAL: WHEN TO USE THIS TOOL vs product_catalog_tool
   ═══════════════════════════════════════════════════════════════════════
   
   ✅ USE spare_parts_pricing_tool FOR:
      • Replacement parts for existing products (handles, cartridges, valves, trims)
      • Component pricing inquiries (e.g., "How much is part TVH.5007?")
      • Spare part numbers starting with: TVL, TVH, TRM, RP, PBV, MEM, K.
      • Parts with format like: 100.1800-2353, K.1800-2229SS
      • When customer asks: "price for replacement handle", "cost of cartridge"
      • Dealer requests for spare parts pricing
   
   ❌ DO NOT USE spare_parts_pricing_tool FOR:
      • Complete product pricing (use product_catalog_tool instead)
      • Product specifications or features
      • Product images or installation guides
      • General product lookups
   
   ═══════════════════════════════════════════════════════════════════════
   
   COMMON SCENARIOS FOR THIS TOOL:
   1. "How much is the replacement cartridge for model 100.1170?"
   2. "Price for TVH.5007 trim ring?"
   3. "Dealer needs pricing for RP70823"
   4. "What does handle 100.1800-2353CP cost?"
   5. "Customer wants to buy replacement valve TRM.PBV.2200BN"
   
   PART NUMBER FORMATS RECOGNIZED:
   • TVL, TVH, TRM prefix (e.g., "TVH.5007", "TRM.TVH.4511CP")
   • RP prefix (e.g., "RP70823", "RP80609MB")
   • Series-based (e.g., "100.1800-2353CP", "K.1800-2229SS")
   • PBV, MEM, K. prefix parts
   
   DATA RETURNED:
   • part_number: Exact part number
   • price: Price string (e.g., "$24.00")
   • price_numeric: Numeric value for calculations
   • price_status: "available" or "not_set" (if "$ -" in database)
   • is_obsolete: True if part is obsolete (may not be available)
   • is_display_dummy: True if display-only (not for sale)
   
   ⚠️ IMPORTANT NOTES:
   • Parts with price_status="not_set" exist but pricing not configured (tell customer to contact sales)
   • Parts marked is_obsolete=True may no longer be available
   • If part not found, check if it's a full product (use product_catalog_tool instead)
   
   PARAM: action_input = {{"part_number": "TVH.5007"}}
          Optional: include_variants=true (to get all finish options for a base part)

10. **finish_tool** [PRIORITY: 10 - MANDATORY]
   - PURPOSE: Complete the agent's reasoning and return all gathered data
   - MANDATORY: You MUST call this to finish (workflow won't end otherwise!)
   - OUTPUT: Structured result for downstream processing
   - PARAM: action_input = {{
        "product_identified": true/false,
        "product_details": {{"model": "...", "name": "...", "category": "..."}},
        "relevant_documents": [...],
        "relevant_images": [...],
        "past_tickets": [...],
        "confidence": 0.0-1.0,
        "reasoning": "Summary of findings"
    }}

═══════════════════════════════════════════════════════════════════════
🎯 OPTIMAL EXECUTION STRATEGY
═══════════════════════════════════════════════════════════════════════

🔴 STEP 0 (MANDATORY IF ATTACHMENTS EXIST): ANALYZE ALL ATTACHMENTS FIRST
  → If customer has PDFs: attachment_analyzer_tool OR multimodal_document_analyzer_tool
  → If customer has images: ocr_image_analyzer_tool
  → Extract: model numbers, part numbers, order numbers, purchase dates, product descriptions
  → CRITICAL: Understand what information you have before selecting next tools

🔴 KEY PRINCIPLE: USE document_search_tool EXTENSIVELY
  → This is your PRIMARY knowledge base with ALL product documentation
  → Search multiple times with different queries to get complete information
  → Always include model number in search when known

IF ATTACHMENTS/IMAGES PRESENT:
  1. ✅ ANALYZE ALL ATTACHMENTS FIRST (attachment_analyzer_tool OR ocr_image_analyzer_tool)
  2. Based on what you found:
     - If found MODEL NUMBERS → product_catalog_tool (verify product)
     - If found ORDER NUMBERS → DO NOT use document_search_tool (it has no order data)
     - If found PART DESCRIPTIONS → document_search_tool with product context
  3. document_search_tool - Search for specs, parts, installation guides
  4. product_catalog_tool - Get pricing, variations, related parts
  5. past_tickets_search_tool
  6. finish_tool

IF TEXT-ONLY QUERY WITH PRODUCT MODEL:
  1. product_catalog_tool (verify product exists, get basic info)
  2. document_search_tool (search "[model] + specific topic" multiple times!)
     - Search for specs: "[model] specifications"
     - Search for parts: "[model] parts diagram"
     - Search for installation: "[model] installation guide"
  3. past_tickets_search_tool (find similar cases)
  4. finish_tool
  
IF SPARE PARTS PRICING REQUEST (replacement handles, cartridges, trims, components):
  ⚠️ This is for COMPONENT/PART pricing, NOT full product pricing!
  
  1. spare_parts_pricing_tool (with the part number)
     - Examples: "TVH.5007", "RP70823", "100.1800-2353CP", "TRM.TVH.4511CP"
     - Returns: price, availability, obsolete status
  2. If part not found → try product_catalog_tool (might be a full product)
  3. finish_tool
  
  RECOGNIZING SPARE PARTS REQUESTS:
  • "How much is part TVH.5007?"
  • "Price for replacement cartridge RP70823"
  • "Cost of trim ring TRM.TVH.4511CP?"
  • "Dealer needs pricing for handles 100.1800-2353"
  • "Customer wants to buy valve PBV.E185-1853"

IF FULL PRODUCT PRICING REQUEST (complete faucets, shower systems, etc.):
  1. product_catalog_tool (has 5,687 complete products with List Price, MAP Price)
  2. finish_tool
  ⚠️ DO NOT use spare_parts_pricing_tool for complete products!

IF DEALER/PARTNERSHIP INQUIRY:
  1. document_search_tool (search for "dealer program", "partnership", "application")
  2. finish_tool

IF INSTALLATION/TROUBLESHOOTING QUERY:
  1. Identify product (from text or images)
  2. document_search_tool - CRITICAL: Search for installation guide
     - "[model] installation manual"
     - "[model] troubleshooting"
     - "[symptom] fix" or "[symptom] solution"
  3. past_tickets_search_tool (similar issues and solutions)
  4. finish_tool

═══════════════════════════════════════════════════════════════════════
⚠️ MANDATORY REQUIREMENTS FOR RETURN/REPLACEMENT/WARRANTY REQUESTS
═══════════════════════════════════════════════════════════════════════

🔴 CRITICAL: Before recommending approval of ANY return, replacement, or warranty claim,
you MUST verify that the customer has provided ALL of the following:

1. ✅ **PO/Purchase Order Number** or proof of purchase
   - Check ticket text and attachments for order numbers, PO numbers, invoices
   
2. ✅ **Video/Photo of the Issue** (for defective products)
   - Check if ticket has image attachments showing the defect/issue
   - If customer claims product is defective but no photo/video provided, flag this!
   
3. ✅ **Shipping Address** (for replacement requests)
   - Check if customer provided a delivery address for the replacement

📌 In your finish_tool, include a "missing_requirements" field listing what's missing:
{{
    "missing_requirements": ["PO number", "photo of defect", "shipping address"]
}}

═══════════════════════════════════════════════════════════════════════
⚠️ CRITICAL RULES (READ CAREFULLY!)
═══════════════════════════════════════════════════════════════════════

✅ DO:
  - 🔴 ANALYZE ALL ATTACHMENTS FIRST before deciding on tools
  - Understand what information each tool contains (see KNOWLEDGE SOURCES section)
  - Call tools in the optimal order
  - Use finish_tool when you've gathered sufficient information
  - If product_catalog_tool returns low confidence or no match, ASSUME the model number is valid and proceed to document_search_tool
  - Pass product context (MODEL NUMBERS) to document_search_tool for better results
  - Check iteration count - are you running out of time?
  - For pricing/dealer/policy questions, go straight to document_search_tool

❌ DON'T:
  - ❌ Use document_search_tool with ORDER NUMBERS (it has no order data!)
  - ❌ Decide which tools to use before analyzing attachments
  - ❌ RETRY the same tool with the same input if it failed once
  - ❌ Get stuck trying to "verify" a product that isn't in the database
  - ❌ Ignore the "This search was already attempted" error message
  - ❌ Forget to call finish_tool (workflow won't complete!)
  - ❌ Force product identification for non-product queries (pricing, dealer inquiries)
  - ❌ Confuse order numbers with product model numbers

🛑 ITERATION LIMIT: {MAX_ITERATIONS} iterations
  - At iteration {MAX_ITERATIONS - 2}: You have ~2 iterations left, start finishing!
  - At iteration {MAX_ITERATIONS - 1}: STOP EVERYTHING, call finish_tool NOW!
  - Iteration {MAX_ITERATIONS}: Forced finish (you're out of time!)

═══════════════════════════════════════════════════════════════════════
📝 RESPONSE FORMAT (JSON ONLY - NO OTHER TEXT)
═══════════════════════════════════════════════════════════════════════

For tool calls:
{{
    "thought": "Step-by-step reasoning of what I know and what I need...",
    "action": "tool_name_exactly_as_listed",
    "action_input": {{"key": "value", "another_key": "value"}}
}}

For finishing:
{{
    "thought": "Summary of reasoning and information gathered...",
    "action": "finish_tool",
    "action_input": {{
        "product_identified": true,
        "product_details": {{
            "model": "100.1170",
            "name": "Delta Faucet Model",
            "category": "Bathroom Faucets"
        }},
        "relevant_documents": [
            {{"title": "Installation Guide", "url": "..."}},
            {{"title": "Warranty Info", "url": "..."}}
        ],
        "relevant_images": ["image_url_1", "image_url_2"],
        "past_tickets": [
            {{"ticket_id": "12345", "resolution": "..."}},
            {{"ticket_id": "12346", "resolution": "..."}}
        ],
        "confidence": 0.92,
        "reasoning": "Found model number in invoice, verified via product search, located installation guide and 3 similar resolved tickets.",
        "missing_requirements": []  // For return/replacement: ["PO number", "photo of defect", "shipping address"]
    }}
}}

═══════════════════════════════════════════════════════════════════════
🔍 DEBUGGING HINTS
═══════════════════════════════════════════════════════════════════════

- If product_search_tool fails, DO NOT RETRY it. Switch to document_search_tool immediately.
- If you have a Model Number but Product Search failed, pass that Model Number to document_search_tool as the "query".
- If you see "This search was already attempted", YOU MUST CHANGE YOUR STRATEGY. Do not repeat the action.
- If iteration count is high (>10), stop searching and call finish_tool.

Remember: Your job is to EFFICIENTLY gather information and call finish_tool.
The downstream processors will use the data you collect to help the customer."""


def react_agent_loop(state: TicketState) -> Dict[str, Any]:
    """
    Main ReACT agent loop with IMPROVED stopping logic.
    Enhanced with Planning Module for better tool orchestration.
    """
    start_time = time.time()
    logger.info(f"{STEP_NAME} | ▶ Starting ReACT agent loop")
    
    ticket_id = state.get("ticket_id", "unknown")
    ticket_subject = state.get("ticket_subject", "")
    ticket_text = state.get("ticket_text", "")
    ticket_images = state.get("ticket_images", [])
    attachments = state.get("ticket_attachments", [])
    
    logger.info(f"{STEP_NAME} | Ticket #{ticket_id}: {len(ticket_text)} chars, {len(ticket_images)} images, {len(attachments)} attachments")
    
    # ========================================
    # PHASE 1: PLANNING MODULE + TICKET FACTS VERIFICATION
    # ========================================
    execution_plan = None
    plan_context = ""
    current_plan_step = 0
    planning_updates = {}
    ticket_facts = state.get("ticket_facts", {}) or {}
    ticket_facts_updates = {}
    constraints_prompt = ""  # NEW: For constraint injection
    constraint_result = None  # NEW: Store constraint validation result
    
    # Run planner if enabled and available
    planner_enabled = getattr(settings, 'enable_planner', True)
    if PLANNER_AVAILABLE and planner_enabled:
        logger.info(f"{STEP_NAME} | 🧠 Running execution planner...")
        try:
            # First verify ticket_facts if not already verified
            if ticket_facts and not ticket_facts.get("planner_verified"):
                logger.info(f"{STEP_NAME} | 🔍 Verifying ticket_facts from extractor...")
                verified_result = verify_ticket_facts(state)
                if verified_result.get("ticket_facts"):
                    ticket_facts = verified_result["ticket_facts"]
                    ticket_facts_updates = {"ticket_facts": ticket_facts}
                    
                    # Log what was verified
                    verified_models = ticket_facts.get("planner_verified_models", [])
                    if verified_models:
                        logger.info(f"{STEP_NAME} | ✅ Verified models: {verified_models}")
            
            # Then create execution plan
            execution_plan = create_execution_plan(state)
            
            if execution_plan and execution_plan.get("execution_plan"):
                plan_context = get_plan_context_for_agent(execution_plan, current_plan_step)
                
                # Extract constraint info from execution plan (NEW)
                if execution_plan.get("_constraint_result"):
                    constraint_result = execution_plan.get("_constraint_result")
                    constraints_prompt = execution_plan.get("_constraints_prompt", "")
                    logger.info(f"{STEP_NAME} | 🔒 Constraints loaded: missing={constraint_result.get('missing_fields', [])}, must_not_ask={len(constraint_result.get('must_not_ask', []))} items")
                
                # Store planning results in state updates
                planning_updates = {
                    "execution_plan": execution_plan,
                    "plan_steps": execution_plan.get("execution_plan", []),
                    "current_plan_step": 0,
                    "ticket_complexity": execution_plan.get("complexity", "moderate"),
                    "planning_confidence": execution_plan.get("confidence", 0.5),
                    "applicable_policy_type": execution_plan.get("policy_applicable", {}).get("policy_type"),
                    "policy_requirements": execution_plan.get("policy_applicable", {}).get("requirements_from_policy", []),
                    "can_proceed_per_policy": execution_plan.get("policy_applicable", {}).get("can_proceed", True),
                    "missing_for_policy": execution_plan.get("policy_applicable", {}).get("missing_for_policy", []),
                    "customer_need_analysis": execution_plan.get("analysis", {}).get("customer_need"),
                    "help_type": execution_plan.get("analysis", {}).get("help_type"),
                    "mentioned_product_model": execution_plan.get("analysis", {}).get("mentioned_product"),
                    # NEW: Add constraint result to state
                    "constraint_result": constraint_result,
                    "constraints_prompt_section": constraints_prompt,
                }
                
                logger.info(f"{STEP_NAME} | ✅ Plan created: {len(execution_plan.get('execution_plan', []))} steps")
                logger.info(f"{STEP_NAME} | Complexity: {execution_plan.get('complexity')}, Policy: {planning_updates.get('applicable_policy_type')}")
            else:
                logger.info(f"{STEP_NAME} | ⚠️ No plan generated, using default strategy")
                
        except Exception as e:
            logger.error(f"{STEP_NAME} | ❌ Planning failed: {e}", exc_info=True)
            # Continue without plan - agent will use default strategy
    else:
        if not PLANNER_AVAILABLE:
            logger.info(f"{STEP_NAME} | Planner module not available")
        elif not planner_enabled:
            logger.info(f"{STEP_NAME} | Planner disabled in settings")
    
    # If no constraints from planner, try to generate them directly (NEW)
    if not constraint_result and CONSTRAINT_VALIDATOR_AVAILABLE:
        try:
            ticket_category = state.get("ticket_category", "") or "general"
            result = validate_constraints(
                ticket_facts=ticket_facts,
                ticket_category=ticket_category,
                product_text=ticket_text[:500]
            )
            constraint_result = result.to_dict()
            constraints_prompt = format_constraints_for_prompt(result)
            planning_updates["constraint_result"] = constraint_result
            planning_updates["constraints_prompt_section"] = constraints_prompt
            logger.info(f"{STEP_NAME} | 🔒 Direct constraints: {format_constraints_summary(result)}")
        except Exception as e:
            logger.warning(f"{STEP_NAME} | ⚠️ Direct constraint validation failed: {e}")
    
    # Log ticket_facts summary for debugging
    if ticket_facts:
        model_candidates = get_model_candidates_from_facts(ticket_facts) if TICKET_EXTRACTOR_AVAILABLE else []
        if model_candidates:
            logger.info(f"{STEP_NAME} | 📦 Model candidates from ticket_facts: {model_candidates}")
        if ticket_facts.get("has_receipt"):
            logger.info(f"{STEP_NAME} | 📄 Receipt/invoice detected in ticket")
        if ticket_facts.get("has_photos"):
            logger.info(f"{STEP_NAME} | 📷 Photos attached to ticket")
    
    # ========================================
    # INITIALIZE AGENT STATE
    # ========================================
    
    logger.info(f"{STEP_NAME} | Ticket #{ticket_id}: {len(ticket_text)} chars, {len(ticket_images)} images, {len(attachments)} attachments")
    
    iterations: List[ReACTIteration] = []
    tool_results = {
        "product_search": None,
        "document_search": None,
        "vision_search": None,
        "past_tickets": None,
        "attachment_analysis": None,
        "attachment_classification": None,
        "multimodal_doc_analysis": None,
        "ocr_image_analysis": None
    }
    
    identified_product = None
    gathered_documents = []
    gathered_images = []
    gathered_past_tickets = []
    product_confidence = 0.0
    gemini_answer = ""
    
    # Vision quality tracking (IMPORTANT for downstream confidence checks)
    vision_match_quality = "NO_MATCH"  # Default
    vision_relevance_reason = ""
    vision_products = []  # Products found specifically via vision_search_tool
    
    # Agent's assessment of missing requirements (from finish_tool)
    agent_missing_requirements = []
    # Image analysis insights (condition, description from OCR analyzer)
    image_analysis_insights = []
    
    llm = get_llm_client()
    
    # Track what we've tried to avoid repetition
    tools_used = set()
    
    for iteration_num in range(1, MAX_ITERATIONS + 1):
        logger.info(f"\n{STEP_NAME} | ═══ ITERATION {iteration_num}/{MAX_ITERATIONS} ═══")
        
        # CRITICAL: Force finish if approaching limit
        if iteration_num >= MAX_ITERATIONS - 1:
            logger.warning(f"{STEP_NAME} | ⚠️ FORCING FINISH - max iterations reached!")
            
            # Build finish tool input from what we have
            finish_input = {
                "product_identified": identified_product is not None,
                "product_details": identified_product or {},
                "relevant_documents": gathered_documents,
                "relevant_images": gathered_images,
                "past_tickets": gathered_past_tickets,
                "confidence": 0.5,
                "reasoning": f"Max iterations ({MAX_ITERATIONS}) reached. Gathered available information."
            }
            
            # Execute finish tool directly
            from app.tools.finish import finish_tool
            if hasattr(finish_tool, "invoke"):
                tool_output = finish_tool.invoke(finish_input)
            elif hasattr(finish_tool, "run"):
                tool_output = finish_tool.run(**finish_input)
            else:
                tool_output = finish_tool._run(**finish_input)
            
            iterations.append({
                "iteration": iteration_num,
                "thought": "Max iterations reached - forcing completion",
                "action": "finish_tool",
                "action_input": finish_input,
                "observation": "Workflow completed",
                "tool_output": tool_output,
                "timestamp": time.time(),
                "duration": 0.0
            })
            
            break
        
        # Build context with clear progress indicators
        agent_context = _build_agent_context(
            ticket_subject=ticket_subject,
            ticket_text=ticket_text,
            ticket_images=ticket_images,
            attachments=attachments,
            iterations=iterations,
            tool_results=tool_results,
            iteration_num=iteration_num,
            max_iterations=MAX_ITERATIONS,
            ticket_facts=ticket_facts  # NEW: Pass ticket_facts as hints
        )
        
        # ========================================
        # INJECT PLAN CONTEXT (Phase 1 Enhancement)
        # ========================================
        if plan_context and execution_plan:
            # Update plan progress
            updated_plan_context = get_plan_context_for_agent(execution_plan, current_plan_step)
            
            # Check if we should suggest next tool from plan
            gathered_info = {
                "product_identified": identified_product is not None,
                "has_attachments": bool(attachments),
                "has_images": bool(ticket_images)
            }
            plan_guidance = should_follow_plan_step(execution_plan, current_plan_step, gathered_info)
            
            # Add plan section to agent context
            plan_section = f"""
═══════════════════════════════════════════════════════════════════════
📋 EXECUTION PLAN (from ticket analysis)
═══════════════════════════════════════════════════════════════════════
{updated_plan_context}

PLAN GUIDANCE: {"Follow suggested tool: " + plan_guidance.get('suggested_tool', '') if plan_guidance.get('follow_plan') else plan_guidance.get('reason', 'Adapt as needed')}

NOTE: You may deviate from the plan based on tool results. The plan is a guide, not a mandate.
═══════════════════════════════════════════════════════════════════════
"""
            # Insert plan context after ticket info
            agent_context = agent_context.replace(
                "═══════════════════════════════════════════════════════════════════════\n📊 PREVIOUS ITERATIONS",
                plan_section + "\n═══════════════════════════════════════════════════════════════════════\n📊 PREVIOUS ITERATIONS"
            )
        
        # ========================================
        # INJECT CONSTRAINTS (Phase 3 Enhancement - NEW)
        # ========================================
        if constraints_prompt:
            # Insert constraints right before PREVIOUS ITERATIONS
            agent_context = agent_context.replace(
                "═══════════════════════════════════════════════════════════════════════\n📊 PREVIOUS ITERATIONS",
                constraints_prompt + "\n═══════════════════════════════════════════════════════════════════════\n📊 PREVIOUS ITERATIONS"
            )
        
        try:
            iteration_start = time.time()
            
            logger.info(f"{STEP_NAME} | 🧠 Calling Gemini for reasoning...")
            response = llm.call_llm(
                system_prompt=REACT_SYSTEM_PROMPT,
                user_prompt=agent_context,
                response_format="json",
                temperature=0.2,  # Lower temperature for more consistent decisions
                max_tokens=settings.llm_max_tokens
            )
            
            if not isinstance(response, dict):
                logger.error(f"{STEP_NAME} | Invalid response format: {response}")
                break
            
            thought = response.get("thought", "")
            action = response.get("action", "")
            action_input = response.get("action_input", {})
            
            logger.info(f"{STEP_NAME} | 💭 Thought: {thought}")
            logger.info(f"{STEP_NAME} | 🔧 Action: {action}")
            logger.info(f"{STEP_NAME} | 📥 Input: {json.dumps(action_input, indent=2)[:200]}")
            
            # Check if trying to repeat a failed tool
            tool_key = f"{action}:{json.dumps(action_input, sort_keys=True)}"
            if tool_key in tools_used and action != "finish_tool":
                logger.warning(f"{STEP_NAME} | ⚠️ Agent trying to repeat tool: {action}")
                observation = "This search was already attempted. Try a different approach or call finish_tool."
                tool_output = {"error": "Duplicate search attempt"}
            else:
                # If the agent calls finish_tool without including the gathered context,
                # inject the already collected resources so downstream nodes get dicts, not bare strings.
                if action == "finish_tool":
                    action_input = dict(action_input or {})

                    # Helper to normalize lists from the LLM (can be strings)
                    def _norm_docs(val):
                        docs = []
                        for d in val or []:
                            if isinstance(d, dict):
                                docs.append(d)
                            elif isinstance(d, str):
                                docs.append({"id": d, "title": d, "content_preview": ""})
                        return docs

                    def _norm_list(val):
                        items = []
                        for x in val or []:
                            if isinstance(x, dict) or isinstance(x, str):
                                items.append(x)
                        return items

                    if not action_input.get("relevant_documents"):
                        action_input["relevant_documents"] = gathered_documents
                    else:
                        action_input["relevant_documents"] = _norm_docs(action_input.get("relevant_documents"))

                    if not action_input.get("relevant_images"):
                        action_input["relevant_images"] = gathered_images
                    else:
                        action_input["relevant_images"] = _norm_list(action_input.get("relevant_images"))

                    if not action_input.get("past_tickets"):
                        action_input["past_tickets"] = gathered_past_tickets
                    else:
                        action_input["past_tickets"] = _norm_list(action_input.get("past_tickets"))

                    if not action_input.get("product_details") and identified_product:
                        action_input["product_details"] = identified_product
                    if "product_identified" not in action_input:
                        action_input["product_identified"] = identified_product is not None
                    if "confidence" not in action_input:
                        action_input["confidence"] = product_confidence or 0.5

                # Execute tool
                tools_used.add(tool_key)
                tool_output, observation = _execute_tool(
                    action=action,
                    action_input=action_input,
                    ticket_images=ticket_images,
                    attachments=attachments,
                    tool_results=tool_results,
                    identified_product=identified_product
                )
            
            iteration_duration = time.time() - iteration_start
            
            logger.info(f"{STEP_NAME} | 📤 Observation: {observation[:200]}...")
            
            # Record iteration
            iteration_record: ReACTIteration = {
                "iteration": iteration_num,
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation,
                "tool_output": tool_output,
                "timestamp": time.time(),
                "duration": iteration_duration
            }
            iterations.append(iteration_record)
            
            # Extract gathered information from tool outputs
            if action == "product_search_tool" and tool_output.get("success"):
                products = tool_output.get("products", [])
                if products and not identified_product:
                    top = products[0]
                    identified_product = {
                        "model": top.get("model_no"),
                        "name": top.get("product_title"),
                        "category": top.get("category"),
                        "confidence": top.get("similarity_score", 0) / 100
                    }
                    product_confidence = identified_product["confidence"]
                    logger.info(f"{STEP_NAME} | ✅ Product identified: {identified_product['model']}")
            
            elif action == "document_search_tool" and tool_output.get("success"):
                docs = tool_output.get("documents", [])
                # Normalize and deduplicate documents by title
                seen_titles = {d.get("title", "").lower() for d in gathered_documents if isinstance(d, dict)}
                for doc in docs:
                    # Ensure doc is a dict
                    if isinstance(doc, str):
                        doc = {"id": doc, "title": doc, "content_preview": ""}
                    elif not isinstance(doc, dict):
                        continue
                    
                    # Deduplicate by title (case-insensitive)
                    doc_title = doc.get("title", "").lower()
                    if doc_title and doc_title not in seen_titles:
                        seen_titles.add(doc_title)
                        gathered_documents.append(doc)
                
                # Store direct Gemini answer for downstream nodes
                if tool_output.get("gemini_answer"):
                    gemini_answer = tool_output.get("gemini_answer", "")
            
            elif action == "vision_search_tool" and tool_output.get("success"):
                matches = tool_output.get("matches", [])
                for match in matches:
                    img_url = match.get("image_url")
                    if img_url and img_url not in gathered_images:
                        gathered_images.append(img_url)
                
                # IMPORTANT: Capture vision match quality for downstream nodes
                vision_match_quality = tool_output.get("match_quality", "NO_MATCH")
                vision_relevance_reason = tool_output.get("reasoning", "")
                logger.info(f"{STEP_NAME} | 🖼️ Vision match quality: {vision_match_quality}")
                
                # Capture ALL vision matches for source_products (Visual Matches section)
                for match in matches:
                    vision_products.append({
                        "model_no": match.get("model_no"),
                        "product_title": match.get("product_title"),
                        "category": match.get("category"),
                        "similarity_score": match.get("similarity_score", 0),
                        "match_level": "🟢" if match.get("similarity_score", 0) >= 85 else "🟡" if match.get("similarity_score", 0) >= 70 else "🔴",
                        "source_type": "vision_search"
                    })
                
                # Vision can also identify product
                if matches and not identified_product:
                    top = matches[0]
                    identified_product = {
                        "model": top.get("model_no"),
                        "name": top.get("product_title"),
                        "category": top.get("category"),
                        "confidence": top.get("similarity_score", 0) / 100
                    }
                    product_confidence = identified_product["confidence"]
            
            elif action == "attachment_analyzer_tool" and tool_output.get("success"):
                # Extract model numbers and other info from attachments
                extracted_info = tool_output.get("extracted_info", {})
                models = extracted_info.get("model_numbers", [])
                if models and not identified_product:
                    # Take first extracted model number
                    logger.info(f"{STEP_NAME} | 📎 Extracted model numbers: {models}")
                    # Note: actual product verification will happen via product_search_tool
            
            elif action == "attachment_type_classifier_tool" and tool_output.get("success"):
                # Categorize attachments for reference
                attachments_classified = tool_output.get("attachments", [])
                logger.info(f"{STEP_NAME} | 📑 Attachment types classified: {len(attachments_classified)} doc(s)")
            
            elif action == "multimodal_document_analyzer_tool" and tool_output.get("success"):
                # Extract complex document data (images within PDFs, tables, etc.)
                docs_analyzed = tool_output.get("documents", [])
                for doc in docs_analyzed:
                    if isinstance(doc, dict):
                        title = doc.get("filename", "Unknown Document")
                        if title not in [d.get("title") for d in gathered_documents if isinstance(d, dict)]:
                            gathered_documents.append({
                                "id": title,
                                "title": title,
                                "content_preview": doc.get("extracted_info", {}).get("text", "")[:500]
                            })
                logger.info(f"{STEP_NAME} | 📄 Multimodal analysis: {len(docs_analyzed)} doc(s) processed")
            
            elif action == "ocr_image_analyzer_tool" and tool_output.get("success"):
                # Extract text from images
                results = tool_output.get("results", [])
                for result in results:
                    img_url = result.get("image_url")
                    if img_url and img_url not in gathered_images:
                        gathered_images.append(img_url)
                
                # Capture image analysis insights (condition, description) for response generation
                # This helps the response generator know if the image shows the defect or not
                if results:
                    image_analysis_insights = []
                    for result in results:
                        insight = {
                            "image_type": result.get("image_type", "unknown"),
                            "description": result.get("description", ""),
                            "condition": result.get("extracted_data", {}).get("condition", ""),
                            "confidence": result.get("confidence", 0)
                        }
                        image_analysis_insights.append(insight)
                    # Store for later use in response generation
                    tool_results["image_analysis_insights"] = image_analysis_insights
                    logger.info(f"{STEP_NAME} | 🖼️  Image condition detected: {image_analysis_insights[0].get('condition', 'unknown')[:50]}...")
                
                logger.info(f"{STEP_NAME} | 🖼️  OCR analysis: {len(results)} image(s) processed")
            
            elif action == "past_tickets_search_tool" and tool_output.get("success"):
                tickets = tool_output.get("tickets", [])
                for ticket in tickets:
                    if ticket not in gathered_past_tickets:
                        gathered_past_tickets.append(ticket)
            
            # ========================================
            # UPDATE PLAN STEP COUNTER (Phase 1)
            # ========================================
            if execution_plan and action != "finish_tool":
                # Check if this action matches the current plan step
                plan_steps = execution_plan.get("execution_plan", [])
                if current_plan_step < len(plan_steps):
                    expected_tool = plan_steps[current_plan_step].get("tool")
                    if action == expected_tool or action.replace("_tool", "") in expected_tool:
                        current_plan_step += 1
                        logger.info(f"{STEP_NAME} | 📋 Plan progress: {current_plan_step}/{len(plan_steps)} steps")
                    else:
                        # Agent deviated from plan - log but don't increment
                        logger.info(f"{STEP_NAME} | 📋 Agent deviated: expected {expected_tool}, got {action}")
            
            # ========================================
            # EARLY TERMINATION CHECK - Stop when answer is found
            # ========================================
            # Detect when we have sufficient evidence to answer the question
            # This prevents redundant iterations after finding the answer
            should_early_terminate = False
            early_terminate_reason = ""
            
            if action != "finish_tool" and iteration_num >= 4:  # Only after a few iterations
                # Count high-quality spec documents
                spec_doc_count = 0
                spec_indicators = [
                    "spec", "specification", "manual", "diagram", "parts",
                    "installation", "output", "diverter", "valve", "cartridge",
                    "pressure", "flow", "gpm", "dimensions"
                ]
                
                for doc in gathered_documents:
                    if isinstance(doc, dict):
                        doc_title = (doc.get("title", "") or "").lower()
                        doc_content = (doc.get("content_preview", "") or "").lower()
                        doc_score = doc.get("relevance_score", 0.5)
                        
                        # Count as spec doc if it contains technical indicators
                        has_specs = any(ind in doc_title or ind in doc_content for ind in spec_indicators)
                        if has_specs and doc_score >= 0.7:
                            spec_doc_count += 1
                
                # Condition 1: Multiple spec documents found for a product inquiry
                # (Agent already has technical specs to answer the question)
                if spec_doc_count >= 3 and identified_product:
                    should_early_terminate = True
                    early_terminate_reason = f"Found {spec_doc_count} specification documents for {identified_product.get('model', 'product')}"
                
                # Condition 2: Document search returned a direct answer (gemini_answer)
                # and we have product context
                elif gemini_answer and len(gemini_answer) > 200 and identified_product:
                    should_early_terminate = True
                    early_terminate_reason = f"Gemini provided comprehensive answer ({len(gemini_answer)} chars) with product context"
                
                # Condition 3: Multiple spec docs found even without catalog match
                # (Product may exist in docs but not in catalog - e.g., PBV.2105)
                elif spec_doc_count >= 4 and len(gathered_documents) >= 5:
                    should_early_terminate = True
                    early_terminate_reason = f"Found {spec_doc_count} specification documents - sufficient for technical inquiry"
                
                # Condition 4: Agent is repeating searches (detected by duplicate attempts)
                # and we already have useful information
                elif "Duplicate search attempt" in str(tool_output) and (spec_doc_count >= 2 or gemini_answer):
                    should_early_terminate = True
                    early_terminate_reason = "Agent repeating searches - proceeding with gathered information"
            
            if should_early_terminate:
                logger.info(f"{STEP_NAME} | 🎯 EARLY TERMINATION: {early_terminate_reason}")
                
                # Build finish tool input with gathered data
                finish_input = {
                    "product_identified": identified_product is not None,
                    "product_details": identified_product or {},
                    "relevant_documents": gathered_documents,
                    "relevant_images": gathered_images,
                    "past_tickets": gathered_past_tickets,
                    "confidence": max(product_confidence, 0.7) if spec_doc_count >= 3 else product_confidence,
                    "reasoning": f"Early termination: {early_terminate_reason}. Gathered {len(gathered_documents)} docs, {len(gathered_past_tickets)} past tickets."
                }
                
                # Execute finish tool directly
                from app.tools.finish import finish_tool
                if hasattr(finish_tool, "invoke"):
                    tool_output = finish_tool.invoke(finish_input)
                elif hasattr(finish_tool, "run"):
                    tool_output = finish_tool.run(**finish_input)
                else:
                    tool_output = finish_tool._run(**finish_input)
                
                iterations.append({
                    "iteration": iteration_num,
                    "thought": f"Early termination triggered: {early_terminate_reason}",
                    "action": "finish_tool",
                    "action_input": finish_input,
                    "observation": "Workflow completed via early termination",
                    "tool_output": tool_output,
                    "timestamp": time.time(),
                    "duration": 0.0
                })
                
                # Update final state
                identified_product = tool_output.get("product_details", identified_product)
                product_confidence = tool_output.get("confidence", product_confidence)
                
                logger.info(f"{STEP_NAME} | ✅ Early termination complete at iteration {iteration_num}")
                break
            
            # Check if finished
            if action == "finish_tool" and tool_output.get("finished"):
                logger.info(f"{STEP_NAME} | ✅ Agent called finish_tool - stopping loop")
                
                # Update from finish tool output
                identified_product = tool_output.get("product_details", identified_product)
                
                # IMPORTANT: Capture missing_requirements from agent's finish_tool input
                # This preserves agent's assessment of what info is still needed
                agent_missing_requirements = action_input.get("missing_requirements", [])
                if agent_missing_requirements:
                    logger.info(f"{STEP_NAME} | 📋 Agent flagged missing requirements: {agent_missing_requirements}")

                # Normalize resources returned by finish_tool - ensure dicts, never strings
                def _normalize_docs(docs):
                    """Normalize documents to always be dicts, deduplicate by title"""
                    if not docs:
                        return []
                    norm = []
                    seen_titles = set()
                    for d in docs:
                        # Convert strings to dicts
                        if isinstance(d, str):
                            d = {"id": d, "title": d, "content_preview": "", "relevance_score": 0.5}
                        elif not isinstance(d, dict):
                            continue
                        
                        # Deduplicate by title (case-insensitive)
                        title = d.get("title", "").lower()
                        if title and title not in seen_titles:
                            seen_titles.add(title)
                            norm.append(d)
                        elif not title:  # Allow docs without titles
                            norm.append(d)
                    return norm

                def _normalize_list(items):
                    """Normalize list items - keep as-is if valid"""
                    if not items:
                        return []
                    norm = []
                    for x in items:
                        if isinstance(x, (dict, str)):
                            norm.append(x)
                    return norm

                # Merge finish_tool results with existing gathered data (prefer existing if better)
                finish_docs = _normalize_docs(tool_output.get("relevant_documents", []))
                # Merge: add finish docs that aren't already in gathered_documents
                existing_titles = {d.get("title", "").lower() for d in gathered_documents if isinstance(d, dict)}
                for doc in finish_docs:
                    title = doc.get("title", "").lower()
                    if title and title not in existing_titles:
                        gathered_documents.append(doc)
                
                finish_images = _normalize_list(tool_output.get("relevant_images", []))
                for img in finish_images:
                    if img not in gathered_images:
                        gathered_images.append(img)
                
                finish_tickets = _normalize_list(tool_output.get("past_tickets", []))
                # Deduplicate tickets by ticket_id
                existing_ticket_ids = {t.get("ticket_id") if isinstance(t, dict) else str(t) for t in gathered_past_tickets}
                for ticket in finish_tickets:
                    ticket_id = ticket.get("ticket_id") if isinstance(ticket, dict) else str(ticket)
                    if ticket_id and ticket_id not in existing_ticket_ids:
                        gathered_past_tickets.append(ticket)
                        existing_ticket_ids.add(ticket_id)
                
                product_confidence = tool_output.get("confidence", product_confidence)
                
                break
                
        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"{STEP_NAME} | ❌ Error in iteration {iteration_num}: {e}", exc_info=True)
            
            # Classify the error type
            if "rate" in error_str and "limit" in error_str:
                error_type = "rate_limit"
            elif "timeout" in error_str or "timed out" in error_str:
                error_type = "timeout"
            elif "quota" in error_str:
                error_type = "quota_exceeded"
            elif "api" in error_str or "connection" in error_str:
                error_type = "api_error"
            else:
                error_type = "internal_error"
            
            # Store error info for downstream handling
            workflow_error = str(e)
            workflow_error_type = error_type
            workflow_error_node = "react_agent"
            is_system_error = True
            
            logger.error(f"{STEP_NAME} | Error type classified as: {error_type}")
            break
    
    # Initialize error tracking variables if not set
    workflow_error = locals().get("workflow_error")
    workflow_error_type = locals().get("workflow_error_type")
    workflow_error_node = locals().get("workflow_error_node")
    is_system_error = locals().get("is_system_error", False)
    
    total_duration = time.time() - start_time
    final_iteration_count = len(iterations)
    
    # Determine status
    if is_system_error:
        status = "error"
    elif final_iteration_count < MAX_ITERATIONS:
        status = "finished"
    else:
        status = "max_iterations"
    
    logger.info(f"\n{STEP_NAME} | ═══ REACT LOOP COMPLETE ═══")
    logger.info(f"{STEP_NAME} | Iterations: {final_iteration_count}/{MAX_ITERATIONS}")
    logger.info(f"{STEP_NAME} | Status: {status}")
    logger.info(f"{STEP_NAME} | Duration: {total_duration:.2f}s")
    logger.info(f"{STEP_NAME} | Product: {identified_product is not None}")
    logger.info(f"{STEP_NAME} | Docs: {len(gathered_documents)}, Images: {len(gathered_images)}, Tickets: {len(gathered_past_tickets)}")
    
    # ========================================
    # SYSTEM ERROR - SKIP ALL DOWNSTREAM PROCESSING
    # ========================================
    if is_system_error:
        logger.error(f"{STEP_NAME} | 🚨 SYSTEM ERROR detected - skipping evidence analysis")
        logger.error(f"{STEP_NAME} | Error: {workflow_error}")
        logger.error(f"{STEP_NAME} | Type: {workflow_error_type}")
        
        # Return immediately with error state - don't run evidence analysis
        audit_events = add_audit_event(
            state,
            event="react_agent_loop",
            event_type="SYSTEM_ERROR",
            details={
                "iterations": final_iteration_count,
                "status": status,
                "duration_seconds": total_duration,
                "workflow_error": workflow_error,
                "workflow_error_type": workflow_error_type,
                "workflow_error_node": workflow_error_node,
            }
        )["audit_events"]
        
        # Still populate legacy fields even on error, so source data is available
        error_legacy_updates = _populate_legacy_fields(
            gathered_documents=gathered_documents,
            gathered_images=gathered_images,
            gathered_past_tickets=gathered_past_tickets,
            identified_product=identified_product,
            product_confidence=product_confidence,
            gemini_answer=gemini_answer,
            vision_products=vision_products,
            spare_parts_pricing=tool_results.get("spare_parts_pricing")  # Include spare parts pricing in context
        )
        
        return {
            "react_iterations": iterations,
            "react_total_iterations": final_iteration_count,
            "react_status": status,
            "react_final_reasoning": f"System error: {workflow_error}",
            "identified_product": identified_product,
            "product_confidence": product_confidence,
            "gathered_documents": gathered_documents,
            "gathered_images": gathered_images,
            "gathered_past_tickets": gathered_past_tickets,
            "gemini_answer": gemini_answer,
            "vision_match_quality": vision_match_quality,
            "vision_relevance_reason": vision_relevance_reason,
            # NO evidence analysis - system error
            "evidence_analysis": {"resolution_action": "error", "final_confidence": 0},
            "needs_more_info": False,  # NOT a "needs more info" situation - it's an ERROR
            "info_request_response": None,
            # Error tracking - this is what draft_response will check
            "workflow_error": workflow_error,
            "workflow_error_type": workflow_error_type,
            "workflow_error_node": workflow_error_node,
            "is_system_error": True,
            # Include ticket_facts even on error (preserve any verification done)
            **(ticket_facts_updates if ticket_facts_updates else {}),
            **error_legacy_updates,  # Include source_products, source_documents, etc.
            "audit_events": audit_events
        }
    
    # ========================================
    # EVIDENCE ANALYSIS (Smart Conflict Resolution)
    # Only run for categories that REQUIRE product identification
    # ========================================
    evidence_analysis = {}
    needs_more_info = False
    info_request_response = None
    
    # Import category groups to check if this ticket needs product identification
    from app.config.constants import NON_PRODUCT_CATEGORIES
    ticket_category = state.get("ticket_category", "general")
    
    # Skip evidence analysis for non-product categories (pricing, dealer inquiries, etc.)
    skip_evidence_check = ticket_category in NON_PRODUCT_CATEGORIES
    
    if skip_evidence_check:
        logger.info(f"{STEP_NAME} | 📋 Category '{ticket_category}' - skipping product evidence check (not product-related)")
        evidence_analysis = {
            "resolution_action": "proceed",
            "final_confidence": 0.5,
            "has_conflict": False,
            "conflict_reason": None,
            "evidence_summary": f"Category '{ticket_category}' does not require product identification",
            "primary_product": None
        }
    elif EVIDENCE_RESOLVER_AVAILABLE:
        logger.info(f"{STEP_NAME} | 🔍 Analyzing evidence for conflicts...")
        
        # Prepare evidence data from tool results
        ocr_result = tool_results.get("ocr_image_analysis")
        vision_result = tool_results.get("vision_search")
        product_results = []
        
        # Collect all product search results from iterations
        for iteration in iterations:
            # Check both product_search_tool and product_catalog_tool
            action = iteration.get("action", "")
            if action in ["product_search_tool", "product_catalog_tool"]:
                output = iteration.get("tool_output", {})
                if output.get("success") and output.get("products"):
                    # Get the source at top level (catalog_cache = exact match)
                    source = output.get("source", "unknown")
                    is_exact = source in ["catalog_cache", "exact", "group"]
                    
                    # Tag each product with the source info so evidence resolver can use it
                    for product in output.get("products", []):
                        product_with_source = product.copy()
                        product_with_source["source"] = source
                        product_with_source["exact_match"] = is_exact
                        product_results.append(product_with_source)
        
        # Analyze evidence - now includes ticket_facts to close the "Split Brain" gap
        ticket_facts = state.get("ticket_facts")
        evidence_bundle = analyze_evidence(
            ocr_result=ocr_result,
            vision_result=vision_result,
            product_search_results=product_results,
            document_results=gathered_documents,
            past_ticket_results=gathered_past_tickets,
            agent_identified_product=identified_product,  # Pass agent's product from finish_tool
            agent_confidence=product_confidence,  # Pass agent's confidence
            ticket_facts=ticket_facts  # Pass pre-extracted ticket facts
        )
        
        logger.info(f"{STEP_NAME} | 📊 Evidence analysis: action={evidence_bundle.resolution_action}, confidence={evidence_bundle.final_confidence:.0%}")
        
        # Check if we need more info from customer
        if evidence_bundle.resolution_action in ["request_info", "escalate"]:
            needs_more_info = True
            requester_name = state.get("requester_name") or "there"
            info_request_response = generate_info_request_response(
                evidence_bundle,
                customer_name=requester_name,
                ticket_subject=state.get("ticket_subject", ""),
                ticket_text=state.get("ticket_text", ""),
                ticket_category=ticket_category,
                ticket_facts=ticket_facts  # Pass ticket_facts to avoid asking for known info
            )
            logger.info(f"{STEP_NAME} | 📝 Generated contextual info request for customer")
        
        elif evidence_bundle.resolution_action == "proceed_with_warning":
            logger.info(f"{STEP_NAME} | ⚠️ Proceeding with warning: {evidence_bundle.conflict_reason}")
        
        # Store evidence analysis for downstream nodes
        evidence_analysis = {
            "resolution_action": evidence_bundle.resolution_action,
            "final_confidence": evidence_bundle.final_confidence,
            "has_conflict": evidence_bundle.has_conflict,
            "conflict_reason": evidence_bundle.conflict_reason,
            "evidence_summary": evidence_bundle.evidence_summary,
            "primary_product": evidence_bundle.primary_product
        }
        
        # If evidence resolver found a better product than what agent identified, use it
        if evidence_bundle.primary_product and evidence_bundle.final_confidence > product_confidence:
            logger.info(f"{STEP_NAME} | 🔄 Evidence resolver updated product to: {evidence_bundle.primary_product.get('model')}")
            identified_product = evidence_bundle.primary_product
            product_confidence = evidence_bundle.final_confidence
    
    if iterations:
        last_thought = iterations[-1]["thought"]
        final_reasoning = f"Completed {final_iteration_count} iterations. {last_thought}"
    else:
        final_reasoning = "No iterations completed"
    
    # Populate legacy fields
    legacy_updates = _populate_legacy_fields(
        gathered_documents=gathered_documents,
        gathered_images=gathered_images,
        gathered_past_tickets=gathered_past_tickets,
        identified_product=identified_product,
        product_confidence=product_confidence,
        gemini_answer=gemini_answer,
        vision_products=vision_products,  # Pass vision-specific products for Visual Matches section
        spare_parts_pricing=tool_results.get("spare_parts_pricing")  # Include spare parts pricing in context
    )
    
    audit_events = add_audit_event(
        state,
        event="react_agent_loop",
        event_type="SUCCESS",
        details={
            "iterations": final_iteration_count,
            "status": status,
            "duration_seconds": total_duration,
            "product_identified": identified_product is not None,
            "documents_found": len(gathered_documents),
            "images_found": len(gathered_images),
            "tickets_found": len(gathered_past_tickets),
            "plan_steps_completed": current_plan_step if execution_plan else 0,
            "plan_total_steps": len(execution_plan.get("execution_plan", [])) if execution_plan else 0,
            "ticket_complexity": planning_updates.get("ticket_complexity") if planning_updates else None,
            "is_system_error": is_system_error,
            "workflow_error": workflow_error,
            "workflow_error_type": workflow_error_type
        }
    )["audit_events"]
    
    # Build final return with planning updates
    result = {
        "react_iterations": iterations,
        "react_total_iterations": final_iteration_count,
        "react_status": status,
        "react_final_reasoning": final_reasoning,
        "identified_product": identified_product,
        "product_confidence": product_confidence,
        "gathered_documents": gathered_documents,
        "gathered_images": gathered_images,
        "gathered_past_tickets": gathered_past_tickets,
        "gemini_answer": gemini_answer,
        # Vision match quality for downstream confidence/hallucination checks
        "vision_match_quality": vision_match_quality,
        "vision_relevance_reason": vision_relevance_reason,
        # Agent's missing requirements assessment (from finish_tool)
        "missing_requirements": agent_missing_requirements,
        # Image analysis insights (condition, description from OCR)
        "image_analysis_insights": tool_results.get("image_analysis_insights", []),
        # Evidence analysis results
        "evidence_analysis": evidence_analysis,
        "needs_more_info": needs_more_info,
        "info_request_response": info_request_response,
        # Error tracking (for proper handling in draft_response)
        "workflow_error": workflow_error,
        "workflow_error_type": workflow_error_type,
        "workflow_error_node": workflow_error_node,
        "is_system_error": is_system_error,
        **legacy_updates,
        "audit_events": audit_events
    }
    
    # Add planning module results if available
    if planning_updates:
        result.update(planning_updates)
        # Update final plan step
        result["current_plan_step"] = current_plan_step
    
    # Add ticket_facts updates if available (from planner verification)
    if ticket_facts_updates:
        result.update(ticket_facts_updates)
    
    return result