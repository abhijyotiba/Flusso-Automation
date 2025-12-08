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

from app.nodes.react_agent_helpers import (
    _build_agent_context,
    _execute_tool,
    _populate_legacy_fields
)

logger = logging.getLogger(__name__)
STEP_NAME = "🤖 REACT_AGENT"

MAX_ITERATIONS = 15

# IMPROVED System Prompt
REACT_SYSTEM_PROMPT = """You are an intelligent support agent helping resolve customer tickets for a plumbing fixtures company.

Your goal: Gather ALL necessary information to help the customer by using available tools strategically.

AVAILABLE TOOLS:
1. **attachment_analyzer_tool** - Extract model numbers from PDFs/invoices (USE FIRST if attachments present!)
2. **product_search_tool** - Search products by model number or description
3. **vision_search_tool** - Identify products from customer images
4. **document_search_tool** - Find installation guides, manuals, FAQs
5. **past_tickets_search_tool** - Find similar resolved tickets
6. **finish_tool** - Submit final context when ready (REQUIRED to complete)

CRITICAL RULES:
1. **ALWAYS call finish_tool when you have gathered information** - This is MANDATORY
2. If you have attachments, START with attachment_analyzer_tool to extract model numbers
3. Once you find a model number, use product_search_tool to verify it exists
4. Use vision_search ONLY if no model number found and customer sent images
5. Search documents AFTER identifying the product
6. Check past tickets ONCE near the end
7. You MUST call finish_tool within {MAX_ITERATIONS} iterations

STOPPING CONDITIONS (call finish_tool when ANY is true):
- ✅ Product identified + found relevant docs/images/tickets
- ✅ Searched all available sources (attachments, text, images)
- ✅ Iteration count >= {MAX_ITERATIONS - 2} (URGENT: finish NOW)
- ✅ Customer query is simple and you have enough basic info

RESPONSE FORMAT (JSON ONLY):
{{
    "thought": "What I know and what I need next...",
    "action": "tool_name",
    "action_input": {{"param": "value"}}
}}

OR when finishing:
{{
    "thought": "I have gathered sufficient information...",
    "action": "finish_tool",
    "action_input": {{
        "product_identified": true/false,
        "product_details": {{"model": "...", "name": "...", "category": "..."}},
        "relevant_documents": [...],
        "relevant_images": [...],
        "past_tickets": [...],
        "confidence": 0.85,
        "reasoning": "Summary of what was found..."
    }}
}}

IMPORTANT: 
- If iteration >= {MAX_ITERATIONS - 2}, you MUST call finish_tool immediately
- Don't repeat failed searches
- Be strategic - prioritize high-value tools first"""


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
    attachments = state.get("attachment_summary", [])
    
    logger.info(f"{STEP_NAME} | Ticket #{ticket_id}: {len(ticket_text)} chars, {len(ticket_images)} images, {len(attachments)} attachments")
    
    iterations: List[ReACTIteration] = []
    tool_results = {
        "product_search": None,
        "document_search": None,
        "vision_search": None,
        "past_tickets": None,
        "attachment_analysis": None
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
            tool_output = finish_tool.invoke(finish_input)
            
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
                max_tokens=2048
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
                for doc in docs:
                    if doc not in gathered_documents:
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

                # Normalize resources returned by finish_tool to dicts/strings we expect
                def _normalize_docs(docs):
                    norm = []
                    for d in docs or []:
                        if isinstance(d, dict):
                            norm.append(d)
                        elif isinstance(d, str):
                            norm.append({"id": d, "title": d, "content_preview": ""})
                    return norm

                def _normalize_list(items):
                    norm = []
                    for x in items or []:
                        if isinstance(x, dict) or isinstance(x, str):
                            norm.append(x)
                    return norm

                gathered_documents = _normalize_docs(tool_output.get("relevant_documents", gathered_documents))
                gathered_images = _normalize_list(tool_output.get("relevant_images", gathered_images))
                gathered_past_tickets = _normalize_list(tool_output.get("past_tickets", gathered_past_tickets))
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