"""
ReACT Agent Node - FIXED VERSION
Reasoning + Acting Loop with proper context tracking
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

logger = logging.getLogger(__name__)
STEP_NAME = "🤖 REACT_AGENT"

MAX_ITERATIONS = 15

# COMPREHENSIVE System Prompt with All Tools
REACT_SYSTEM_PROMPT = """You are an intelligent support agent helping resolve customer tickets for a plumbing fixtures company.

Your goal: Gather ALL necessary information to help the customer by using available tools strategically and efficiently.

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
   - PURPOSE: Deep analysis of documents using vision + text (images within PDFs)
   - USE WHEN: Documents have diagrams, charts, or complex visual layouts
   - OUTPUT: Extracted text, images, tables, structural analysis
   - PARAM: action_input = {{"attachments": [...]}}

4. **ocr_image_analyzer_tool** [PRIORITY: 4]
   - PURPOSE: Extract text from customer's uploaded images (photos, screenshots)
   - USE WHEN: Customer sent photos of products, labels, or error messages
   - OUTPUT: Extracted text, identified model numbers, error codes
   - PARAM: action_input = {{"image_urls": [...]}}

5. **product_search_tool** [PRIORITY: 5⭐]
   - PURPOSE: Find products by model number, name, or description
   - CRITICAL: Always verify extracted model numbers with this tool
   - OUTPUT: Product details, model info, category, specifications
   - PARAM: action_input = {{"query": "search term"}} OR {{"model_number": "MODEL_NO"}}

6. **vision_search_tool** [PRIORITY: 6]
   - PURPOSE: Identify products from customer images using vision AI
   - USE WHEN: No model number found but customer sent images
   - OUTPUT: Identified product, match quality, confidence score
   - PARAM: action_input = {{"image_urls": [...]}}

7. **document_search_tool** [PRIORITY: 7⭐]
   - PURPOSE: Find installation guides, manuals, FAQs, technical docs
   - BEST USED AFTER: Product identified (use product context)
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

IF ATTACHMENTS/IMAGES PRESENT:
  1. attachment_analyzer_tool OR ocr_image_analyzer_tool (extract info)
  2. product_search_tool (verify extracted model numbers)
  3. [CRITICAL BRANCH]
     - IF product_search finds a match: -> document_search_tool (using product name)
     - IF product_search FAILS/LOW CONFIDENCE: -> document_search_tool (using extracted model # directly)
  4. past_tickets_search_tool
  5. finish_tool

IF TEXT-ONLY QUERY:
  1. product_search_tool (search for product by description)
  2. document_search_tool (find relevant docs)
  3. past_tickets_search_tool (find similar cases)
  4. finish_tool

═══════════════════════════════════════════════════════════════════════
⚠️ CRITICAL RULES (READ CAREFULLY!)
═══════════════════════════════════════════════════════════════════════

✅ DO:
  - Call tools in the optimal order
  - Use finish_tool when you've gathered sufficient information
  - If product_search_tool returns low confidence or no match, ASSUME the model number is valid and proceed to document_search_tool.
  - Pass product context to document_search for better results
  - Check iteration count - are you running out of time?

❌ DON'T:
  - RETRY the same tool with the same input if it failed once.
  - Get stuck trying to "verify" a product that isn't in the database.
  - Ignore the "This search was already attempted" error message.
  - Forget to call finish_tool (workflow won't complete!)

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
    Main ReACT agent loop with IMPROVED stopping logic
    """
    start_time = time.time()
    logger.info(f"{STEP_NAME} | ▶ Starting ReACT agent loop")
    
    ticket_id = state.get("ticket_id", "unknown")
    ticket_subject = state.get("ticket_subject", "")
    ticket_text = state.get("ticket_text", "")
    ticket_images = state.get("ticket_images", [])
    attachments = state.get("ticket_attachments", [])
    
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
            logger.error(f"{STEP_NAME} | ❌ Error in iteration {iteration_num}: {e}", exc_info=True)
            break
    
    total_duration = time.time() - start_time
    final_iteration_count = len(iterations)
    
    status = "finished" if final_iteration_count < MAX_ITERATIONS else "max_iterations"
    
    logger.info(f"\n{STEP_NAME} | ═══ REACT LOOP COMPLETE ═══")
    logger.info(f"{STEP_NAME} | Iterations: {final_iteration_count}/{MAX_ITERATIONS}")
    logger.info(f"{STEP_NAME} | Status: {status}")
    logger.info(f"{STEP_NAME} | Duration: {total_duration:.2f}s")
    logger.info(f"{STEP_NAME} | Product: {identified_product is not None}")
    logger.info(f"{STEP_NAME} | Docs: {len(gathered_documents)}, Images: {len(gathered_images)}, Tickets: {len(gathered_past_tickets)}")
    
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
            "tickets_found": len(gathered_past_tickets)
        }
    )["audit_events"]
    
    return {
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
        **legacy_updates,
        "audit_events": audit_events
    }