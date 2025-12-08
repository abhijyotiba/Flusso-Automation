"""
ReACT Agent Node - Reasoning + Acting Loop
Uses Gemini 2.0 Flash Pro to intelligently gather information
"""

import logging
import time
import json
from typing import Dict, Any, List

from app.graph.state import TicketState, ReACTIteration
from app.clients.llm_client import get_llm_client
from app.utils.audit import add_audit_event

# Import helper functions
from app.nodes.react_agent_helpers import (
    _build_agent_context,
    _execute_tool,
    _populate_legacy_fields
)

logger = logging.getLogger(__name__)
STEP_NAME = "🤖 REACT_AGENT"

# Maximum iterations to prevent infinite loops
MAX_ITERATIONS = 15

# ReACT System Prompt
REACT_SYSTEM_PROMPT = """You are an intelligent support agent helping resolve customer tickets for a plumbing fixtures company.

Your goal: Gather ALL necessary information to help the customer by using available tools strategically.

AVAILABLE TOOLS:
1. product_search_tool - Search products by model number or description (Pinecone metadata)
2. document_search_tool - Find installation guides, manuals, FAQs (Gemini File Search)
3. vision_search_tool - Identify products from customer images (CLIP visual similarity)
4. past_tickets_search_tool - Find similar resolved tickets (historical patterns)
5. attachment_analyzer_tool - Extract info from PDFs/invoices (model numbers, order details)
6. finish_tool - Submit final context when ready (REQUIRED to complete)

REASONING PROCESS (ReACT):
Each iteration, you must:
1. **Thought**: Analyze what you know and what's missing
2. **Action**: Choose ONE tool to use next
3. **Action Input**: Provide parameters for the tool
4. **Observation**: Receive tool output
5. **Repeat** until you have enough information OR max iterations reached

STRATEGIC GUIDELINES:
- Start with attachment_analyzer if PDFs/documents present (may contain model numbers)
- Use product_search FIRST if model number is mentioned in text
- Use vision_search if customer attached product images
- Only search documents AFTER you know the product
- Check past_tickets once you've identified the product
- Call finish_tool when you have sufficient context

PRODUCT IDENTIFICATION PRIORITY:
1. Exact model number in text → product_search_tool(model_number="...")
2. Model in PDF attachment → attachment_analyzer_tool → then product_search
3. Customer image → vision_search_tool → validate with product_search
4. Vague description → product_search_tool(query="...") → narrow down

STOPPING CONDITIONS:
- You MUST call finish_tool when:
  a) Product identified + found relevant docs/images/tickets
  b) Searched all available sources
  c) Iteration count approaching maximum
  
You MUST respond in this EXACT JSON format:
{
    "thought": "What I know and what I need next...",
    "action": "tool_name",
    "action_input": {
        "param1": "value1",
        "param2": "value2"
    }
}

OR when ready to finish:
{
    "thought": "I have gathered sufficient information...",
    "action": "finish_tool",
    "action_input": {
        "product_identified": true/false,
        "product_details": {"model": "...", "name": "...", "category": "..."},
        "relevant_documents": [...],
        "relevant_images": [...],
        "past_tickets": [...],
        "confidence": 0.85,
        "reasoning": "Explain what you found..."
    }
}

CRITICAL RULES:
- ONE tool per iteration
- ALWAYS call finish_tool before max iterations
- Be strategic - don't repeat failed searches
- Cross-validate: vision results → confirm with product_search
- Prioritize accuracy over speed"""


def react_agent_loop(state: TicketState) -> Dict[str, Any]:
    """
    Main ReACT agent loop - iteratively gathers information
    
    Returns:
        Updated state with:
        - react_iterations
        - react_status
        - identified_product
        - gathered_documents
        - gathered_images
        - gathered_past_tickets
    """
    start_time = time.time()
    logger.info(f"{STEP_NAME} | ▶ Starting ReACT agent loop")
    
    # Extract ticket info
    ticket_id = state.get("ticket_id", "unknown")
    ticket_subject = state.get("ticket_subject", "")
    ticket_text = state.get("ticket_text", "")
    ticket_images = state.get("ticket_images", [])
    attachments = state.get("attachment_summary", [])
    
    logger.info(f"{STEP_NAME} | Ticket #{ticket_id}: {len(ticket_text)} chars text, {len(ticket_images)} images, {len(attachments)} attachments")
    
    # Initialize ReACT state
    iterations: List[ReACTIteration] = []
    tool_results = {
        "product_search": None,
        "document_search": None,
        "vision_search": None,
        "past_tickets": None,
        "attachment_analysis": None
    }
    
    # Track what we've gathered
    identified_product = None
    gathered_documents = []
    gathered_images = []
    gathered_past_tickets = []
    product_confidence = 0.0
    
    llm = get_llm_client()
    
    # Main ReACT loop
    for iteration_num in range(1, MAX_ITERATIONS + 1):
        logger.info(f"\n{STEP_NAME} | ═══ ITERATION {iteration_num}/{MAX_ITERATIONS} ═══")
        
        # Build context for agent
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
        
        # Call Gemini to reason about next action
        try:
            iteration_start = time.time()
            
            logger.info(f"{STEP_NAME} | 🧠 Calling Gemini 2.5 Flash Pro for reasoning...")
            response = llm.call_llm(
                system_prompt=REACT_SYSTEM_PROMPT,
                user_prompt=agent_context,
                response_format="json",
                temperature=0.3,  # Balance creativity with consistency
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
            logger.info(f"{STEP_NAME} | 📥 Input: {json.dumps(action_input, indent=2)}")
            
            # Execute tool
            tool_output, observation = _execute_tool(
                action=action,
                action_input=action_input,
                ticket_images=ticket_images,
                attachments=attachments,
                tool_results=tool_results
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
            
            # Check if finished
            if action == "finish_tool" and tool_output.get("finished"):
                logger.info(f"{STEP_NAME} | ✅ Agent called finish_tool - stopping loop")
                
                # Extract final results
                identified_product = tool_output.get("product_details")
                gathered_documents = tool_output.get("relevant_documents", [])
                gathered_images = tool_output.get("relevant_images", [])
                gathered_past_tickets = tool_output.get("past_tickets", [])
                product_confidence = tool_output.get("confidence", 0.5)
                
                break
                
        except Exception as e:
            logger.error(f"{STEP_NAME} | ❌ Error in iteration {iteration_num}: {e}", exc_info=True)
            break
    
    # Calculate final metrics
    total_duration = time.time() - start_time
    final_iteration_count = len(iterations)
    
    status = "finished" if final_iteration_count < MAX_ITERATIONS else "max_iterations"
    
    logger.info(f"\n{STEP_NAME} | ═══ REACT LOOP COMPLETE ═══")
    logger.info(f"{STEP_NAME} | Iterations: {final_iteration_count}/{MAX_ITERATIONS}")
    logger.info(f"{STEP_NAME} | Status: {status}")
    logger.info(f"{STEP_NAME} | Duration: {total_duration:.2f}s")
    logger.info(f"{STEP_NAME} | Product identified: {identified_product is not None}")
    logger.info(f"{STEP_NAME} | Documents: {len(gathered_documents)}, Images: {len(gathered_images)}, Tickets: {len(gathered_past_tickets)}")
    
    # Build final reasoning
    if iterations:
        last_thought = iterations[-1]["thought"]
        final_reasoning = f"Completed {final_iteration_count} iterations. {last_thought}"
    else:
        final_reasoning = "No iterations completed"
    
    # Populate legacy fields for compatibility with existing nodes
    legacy_updates = _populate_legacy_fields(
        gathered_documents=gathered_documents,
        gathered_images=gathered_images,
        gathered_past_tickets=gathered_past_tickets,
        identified_product=identified_product,
        product_confidence=product_confidence
    )
    
    # Build audit event
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
        **legacy_updates,  # Populate legacy RAG result fields
        "audit_events": audit_events
    }
