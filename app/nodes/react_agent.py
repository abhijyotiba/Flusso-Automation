"""
ReACT Agent Node - FIXED VERSION
Reasoning + Acting Loop with proper context tracking
Enhanced with Planning Module (Phase 1)
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
        should_follow_plan_step
    )
    PLANNER_AVAILABLE = True
except ImportError:
    PLANNER_AVAILABLE = False
    logging.getLogger(__name__).warning("Planner module not available")

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

1. **Gemini File Search (document_search_tool)** 📖
   ✅ HAS: Product specification PDFs with technical diagrams
   ✅ HAS: Parts diagrams showing product components and specific parts
   ✅ HAS: Parts lists with part numbers and MSRP/pricing
   ✅ HAS: Installation guides and manuals
   ✅ HAS: Core policy documents:
      - Warranty policy
      - MAP (Minimum Advertised Price) agreement
      - Training manuals
      - Return/refund policies
   ✅ HAS: Troubleshooting guides and FAQs
   ❌ DOES NOT HAVE: Customer order data, purchase orders, invoices, shipping/tracking info
   ❌ DOES NOT HAVE: Customer account information or order history
   
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

═══════════════════════════════════════════════════════════════════════
🎯 TICKET TYPE HANDLING - DIFFERENT APPROACHES FOR DIFFERENT QUERIES
═══════════════════════════════════════════════════════════════════════

**PRODUCT ISSUES** (defects, broken, warranty, replacement parts):
  → Use vision/OCR to identify product
  → Verify with product_search
  → Find relevant docs and past tickets

**PRICING/MSRP REQUESTS** (asking for price of parts):
  → Use document_search_tool to find pricing information
  → Search with part numbers directly
  → NO need for product identification or vision

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

7. **document_search_tool** [PRIORITY: 7⭐]
   - PURPOSE: Find installation guides, manuals, FAQs, technical docs, PRICING, POLICIES
   - BEST USED AFTER: Product identified (use product context)
   - ALSO: Use for pricing queries, dealer info, warranty policies
   - OUTPUT: Relevant documentation, snippets, installation steps
   - PARAM: action_input = {{"query": "search term", "product_context": "product_name"}}

8. **past_tickets_search_tool** [PRIORITY: 8]
   - PURPOSE: Find similar resolved tickets and solutions
   - USE NEAR END: After identifying product, before finishing
   - OUTPUT: Similar tickets, resolutions, common solutions, patterns
   - PARAM: action_input = {{"query": "search term"}}

9. **finish_tool** [PRIORITY: 9 - MANDATORY]
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

IF ATTACHMENTS/IMAGES PRESENT:
  1. ✅ ANALYZE ALL ATTACHMENTS FIRST (attachment_analyzer_tool OR ocr_image_analyzer_tool)
  2. Based on what you found:
     - If found MODEL NUMBERS → product_catalog_tool (verify product)
     - If found ORDER NUMBERS → DO NOT use document_search_tool (it has no order data)
     - If found PART DESCRIPTIONS → document_search_tool with product context
  3. product_catalog_tool or document_search_tool (based on findings)
  4. past_tickets_search_tool
  5. finish_tool

IF TEXT-ONLY QUERY:
  1. product_catalog_tool (search for product by description)
  2. document_search_tool (find relevant docs)
  3. past_tickets_search_tool (find similar cases)
  4. finish_tool
  
IF PRICING/INFORMATION REQUEST (no product ID needed):
  1. document_search_tool (search for the specific info requested)
  2. past_tickets_search_tool (find similar inquiries)
  3. finish_tool

IF DEALER/PARTNERSHIP INQUIRY:
  1. document_search_tool (search for "dealer program", "partnership", "application")
  2. finish_tool

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
        "reasoning": "Found model number in invoice, verified via product search, located installation guide and 3 similar resolved tickets."
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
    # PHASE 1: PLANNING MODULE
    # ========================================
    execution_plan = None
    plan_context = ""
    current_plan_step = 0
    planning_updates = {}
    
    # Run planner if enabled and available
    planner_enabled = getattr(settings, 'enable_planner', True)
    if PLANNER_AVAILABLE and planner_enabled:
        logger.info(f"{STEP_NAME} | 🧠 Running execution planner...")
        try:
            execution_plan = create_execution_plan(state)
            
            if execution_plan and execution_plan.get("execution_plan"):
                plan_context = get_plan_context_for_agent(execution_plan, current_plan_step)
                
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
            max_iterations=MAX_ITERATIONS
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
            
            # Check if finished
            if action == "finish_tool" and tool_output.get("finished"):
                logger.info(f"{STEP_NAME} | ✅ Agent called finish_tool - stopping loop")
                
                # Update from finish tool output
                identified_product = tool_output.get("product_details", identified_product)

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
        
        # Analyze evidence
        evidence_bundle = analyze_evidence(
            ocr_result=ocr_result,
            vision_result=vision_result,
            product_search_results=product_results,
            document_results=gathered_documents,
            past_ticket_results=gathered_past_tickets,
            agent_identified_product=identified_product,  # Pass agent's product from finish_tool
            agent_confidence=product_confidence  # Pass agent's confidence
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
                ticket_category=ticket_category
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
        gemini_answer=gemini_answer
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
    
    return result