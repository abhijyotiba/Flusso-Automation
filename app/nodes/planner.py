"""
Ticket Analysis & Planning Module
Analyzes ticket and creates structured execution plan BEFORE tool calls.
Integrates with policy service for policy-aware planning.
"""

import logging
import time
from typing import Dict, Any, List, Optional

from app.graph.state import TicketState
from app.clients.llm_client import get_llm_client
from app.services.policy_service import get_relevant_policy, get_policy_for_category
from app.config.settings import settings
from app.utils.audit import add_audit_event

logger = logging.getLogger(__name__)
STEP_NAME = "🧠 PLANNER"


# ===============================
# PLANNING PROMPT
# ===============================
PLANNING_PROMPT = """You are a support ticket analysis expert for Flusso Kitchen & Bath (plumbing fixtures company).

Your job is to:
1. Analyze the customer ticket
2. Understand company policies that apply
3. Create an optimal execution plan for gathering information

═══════════════════════════════════════════════════════════════════════
📋 TICKET INFORMATION
═══════════════════════════════════════════════════════════════════════
Subject: {subject}
Description: {text}
Category: {category}
Has Images: {has_images} ({image_count} images)
Has Attachments: {has_attachments} ({attachment_count} attachments)
Attachment Types: {attachment_types}

═══════════════════════════════════════════════════════════════════════
📜 RELEVANT COMPANY POLICY
═══════════════════════════════════════════════════════════════════════
{policy_section}

Policy Requirements for this type of request:
{policy_requirements}

═══════════════════════════════════════════════════════════════════════
🔧 AVAILABLE TOOLS (use exact names)
═══════════════════════════════════════════════════════════════════════
1. attachment_analyzer_tool - Extract data from PDFs, invoices, receipts, docs
   USE WHEN: Customer attached PDF/document that may contain purchase info, model numbers

2. ocr_image_analyzer_tool - Extract text from images (photos of labels, receipts)
   USE WHEN: Customer sent photos that may have text/model numbers visible

3. product_search_tool - Find products by model number or description
   USE WHEN: Need to verify a product exists, get product details

4. vision_search_tool - Identify products from images using visual similarity
   USE WHEN: Customer sent product photo but no model number mentioned

5. document_search_tool - Search manuals, FAQs, installation guides, troubleshooting
   USE WHEN: Need installation help, troubleshooting steps, product specifications

6. past_tickets_search_tool - Find similar resolved tickets
   USE WHEN: Want to see how similar issues were resolved before

7. finish_tool - Complete the information gathering process
   ALWAYS END WITH THIS

═══════════════════════════════════════════════════════════════════════
📝 YOUR TASK
═══════════════════════════════════════════════════════════════════════

Analyze this ticket and create a structured plan. Consider:

1. WHAT does the customer need? (warranty claim, return, parts, help, info?)
2. WHAT product are they asking about? (model number mentioned?)
3. WHAT documents/proof do they need per policy? (receipt, photos?)
4. WHAT tools should we use and in what ORDER?

IMPORTANT RULES:
- If policy requires proof of purchase → check attachments FIRST
- If customer mentions model number → verify with product_search
- If customer has images but no model → try OCR first, then vision
- Always include document_search for troubleshooting/installation tickets
- Consider past_tickets for common issues
- ALWAYS end plan with finish_tool

═══════════════════════════════════════════════════════════════════════

Respond ONLY with valid JSON in this exact format:
{{
    "analysis": {{
        "customer_need": "Brief description of what customer wants",
        "mentioned_product": "Model number if mentioned, or null",
        "help_type": "warranty|return|parts|installation|troubleshooting|inquiry|general",
        "urgency": "low|medium|high",
        "key_details": ["detail1", "detail2"]
    }},
    "policy_applicable": {{
        "policy_type": "warranty|return|replacement|missing_parts|none",
        "requirements_from_policy": ["requirement1", "requirement2"],
        "can_proceed": true,
        "missing_for_policy": ["what's missing if any"]
    }},
    "information_needs": {{
        "product_identification": true/false,
        "proof_of_purchase": true/false,
        "installation_docs": true/false,
        "troubleshooting_info": true/false,
        "warranty_info": true/false,
        "past_ticket_patterns": true/false,
        "customer_photos_analysis": true/false
    }},
    "execution_plan": [
        {{"step": 1, "tool": "tool_name", "reason": "why this tool", "input_hint": "what to search/analyze"}},
        {{"step": 2, "tool": "tool_name", "reason": "why", "input_hint": "hint"}},
        ...
        {{"step": N, "tool": "finish_tool", "reason": "compile findings", "input_hint": null}}
    ],
    "complexity": "simple|moderate|complex",
    "estimated_tools": 3,
    "confidence": 0.85
}}
"""


# ===============================
# HELPER FUNCTIONS
# ===============================
def _extract_model_numbers(text: str) -> List[str]:
    """Extract potential model numbers from text."""
    import re
    
    # Common patterns for model numbers
    patterns = [
        r'\b\d{3}\.\d{4}(?:-[A-Z]+)?\b',  # 100.1170, 100.1170-PC
        r'\b[A-Z]{2}\d{4}[A-Z]{0,2}\b',    # HS6270MB, HS1006
        r'\b\d{3}\.\d{4}-\d{4}\b',          # 160.1168-9862
        r'\b[A-Z]{2,3}-?\d{3,4}-?[A-Z]{0,3}\b',  # HS-6270-MB
    ]
    
    found = []
    for pattern in patterns:
        matches = re.findall(pattern, text.upper())
        found.extend(matches)
    
    return list(set(found))


def _quick_classify(state: TicketState) -> str:
    """Quick classification based on keywords for policy lookup."""
    text = (state.get("ticket_subject", "") + " " + state.get("ticket_text", "")).lower()
    
    # Check for warranty indicators
    warranty_words = ["warranty", "defect", "broken", "not working", "stopped working", "malfunction", "leaking"]
    if any(w in text for w in warranty_words):
        return "warranty"
    
    # Check for return indicators
    return_words = ["return", "refund", "rga", "send back", "money back"]
    if any(w in text for w in return_words):
        return "return"
    
    # Check for missing parts
    missing_words = ["missing", "not included", "incomplete", "didn't receive"]
    if any(w in text for w in missing_words):
        return "missing_parts"
    
    # Check for replacement parts
    parts_words = ["replacement", "spare part", "need part", "part number"]
    if any(w in text for w in parts_words):
        return "replacement_parts"
    
    # Check for installation
    install_words = ["install", "installation", "how to", "setup", "mounting"]
    if any(w in text for w in install_words):
        return "installation"
    
    # Check for shipping
    shipping_words = ["shipping", "tracking", "delivery", "where is", "when will"]
    if any(w in text for w in shipping_words):
        return "shipping"
    
    return "general"


def _get_attachment_types(attachments: List[Dict]) -> str:
    """Get human-readable attachment types."""
    if not attachments:
        return "None"
    
    types = []
    for att in attachments:
        filename = att.get("filename", att.get("name", "")).lower()
        if filename.endswith(".pdf"):
            types.append("PDF")
        elif filename.endswith((".jpg", ".jpeg", ".png", ".gif")):
            types.append("Image")
        elif filename.endswith((".doc", ".docx")):
            types.append("Document")
        else:
            types.append("Other")
    
    return ", ".join(set(types)) if types else "Unknown"


def _build_default_plan(state: TicketState, policy_type: str) -> Dict[str, Any]:
    """Build a default plan when LLM fails."""
    has_images = bool(state.get("ticket_images", []))
    has_attachments = bool(state.get("ticket_attachments", []))
    text = state.get("ticket_text", "")
    
    # Extract model numbers
    model_numbers = _extract_model_numbers(text)
    
    plan_steps = []
    step = 1
    
    # Priority 1: Process attachments if present (may have receipt/proof)
    if has_attachments:
        plan_steps.append({
            "step": step,
            "tool": "attachment_analyzer_tool",
            "reason": "Extract info from attachments (may contain proof of purchase, model numbers)",
            "input_hint": "focus on dates, model numbers, order info"
        })
        step += 1
    
    # Priority 2: OCR images if present
    if has_images:
        plan_steps.append({
            "step": step,
            "tool": "ocr_image_analyzer_tool",
            "reason": "Extract text from customer images (may have labels, model numbers)",
            "input_hint": None
        })
        step += 1
    
    # Priority 3: Product search if model mentioned
    if model_numbers:
        plan_steps.append({
            "step": step,
            "tool": "product_search_tool",
            "reason": f"Verify product model: {model_numbers[0]}",
            "input_hint": model_numbers[0]
        })
        step += 1
    elif has_images:
        # Vision search if no model but has images
        plan_steps.append({
            "step": step,
            "tool": "vision_search_tool",
            "reason": "Identify product from customer images",
            "input_hint": None
        })
        step += 1
    
    # Priority 4: Document search for relevant info
    plan_steps.append({
        "step": step,
        "tool": "document_search_tool",
        "reason": "Find relevant documentation (manuals, troubleshooting, warranty)",
        "input_hint": model_numbers[0] if model_numbers else "product documentation"
    })
    step += 1
    
    # Priority 5: Past tickets for patterns
    plan_steps.append({
        "step": step,
        "tool": "past_tickets_search_tool",
        "reason": "Find similar resolved tickets",
        "input_hint": text[:100] if text else "similar issues"
    })
    step += 1
    
    # Always finish
    plan_steps.append({
        "step": step,
        "tool": "finish_tool",
        "reason": "Compile all gathered information",
        "input_hint": None
    })
    
    return {
        "analysis": {
            "customer_need": "Unable to analyze - using default plan",
            "mentioned_product": model_numbers[0] if model_numbers else None,
            "help_type": policy_type,
            "urgency": "medium",
            "key_details": []
        },
        "policy_applicable": {
            "policy_type": policy_type,
            "requirements_from_policy": [],
            "can_proceed": True,
            "missing_for_policy": []
        },
        "information_needs": {
            "product_identification": True,
            "proof_of_purchase": policy_type in ["warranty", "return"],
            "installation_docs": policy_type == "installation",
            "troubleshooting_info": True,
            "warranty_info": policy_type == "warranty",
            "past_ticket_patterns": True,
            "customer_photos_analysis": has_images
        },
        "execution_plan": plan_steps,
        "complexity": "moderate",
        "estimated_tools": len(plan_steps),
        "confidence": 0.5
    }


# ===============================
# MAIN PLANNING FUNCTION
# ===============================
def create_execution_plan(state: TicketState) -> Dict[str, Any]:
    """
    Analyze ticket and create execution plan before ReACT loop.
    
    This function:
    1. Analyzes the ticket content
    2. Fetches relevant policy based on ticket type
    3. Creates an ordered execution plan for tools
    
    Args:
        state: Current ticket state
        
    Returns:
        Dict containing:
        - analysis: Ticket analysis
        - policy_applicable: What policies apply
        - information_needs: What info to gather
        - execution_plan: Ordered list of tool calls
        - complexity: simple/moderate/complex
        - confidence: Planning confidence score
    """
    start_time = time.time()
    logger.info(f"{STEP_NAME} | ▶ Starting execution planning...")
    
    # Check if planner is enabled
    if hasattr(settings, 'enable_planner') and not settings.enable_planner:
        logger.info(f"{STEP_NAME} | Planner disabled, skipping")
        return {}
    
    # Extract ticket info
    subject = state.get("ticket_subject", "") or ""
    text = state.get("ticket_text", "") or ""
    category = state.get("ticket_category", "") or "general"
    images = state.get("ticket_images", []) or []
    attachments = state.get("ticket_attachments", []) or []
    
    logger.info(f"{STEP_NAME} | Ticket: '{subject[:50]}...', Category: {category}")
    logger.info(f"{STEP_NAME} | Images: {len(images)}, Attachments: {len(attachments)}")
    
    # Step 1: Quick classify for policy lookup
    quick_category = _quick_classify(state)
    logger.info(f"{STEP_NAME} | Quick classification: {quick_category}")
    
    # Step 2: Get relevant policy
    policy_result = get_relevant_policy(
        ticket_category=category or quick_category,
        ticket_text=text,
        keywords=_extract_model_numbers(text)
    )
    
    policy_section = policy_result.get("primary_section", "")[:2000]  # Limit for prompt
    policy_requirements = policy_result.get("policy_requirements", [])
    
    logger.info(f"{STEP_NAME} | Policy section: {policy_result.get('primary_section_name', 'N/A')}")
    logger.info(f"{STEP_NAME} | Policy requirements: {policy_requirements}")
    
    # Step 3: Build prompt
    prompt = PLANNING_PROMPT.format(
        subject=subject,
        text=text[:1500],  # Limit text length
        category=category or "not classified",
        has_images="Yes" if images else "No",
        image_count=len(images),
        has_attachments="Yes" if attachments else "No",
        attachment_count=len(attachments),
        attachment_types=_get_attachment_types(attachments),
        policy_section=policy_section or "No specific policy found",
        policy_requirements="\n".join(f"- {req}" for req in policy_requirements) if policy_requirements else "No specific requirements"
    )
    
    # Step 4: Call LLM for planning
    try:
        llm = get_llm_client()
        
        logger.info(f"{STEP_NAME} | 🔄 Calling LLM for planning...")
        llm_start = time.time()
        
        response = llm.call_llm(
            system_prompt="You are a ticket analysis and planning expert. Respond only with valid JSON. Keep responses concise.",
            user_prompt=prompt,
            response_format="json",
            temperature=0.1,  # Low temperature for consistent planning
            max_tokens=4096  # Increased from 2048 to avoid truncation
        )
        
        llm_duration = time.time() - llm_start
        logger.info(f"{STEP_NAME} | ✓ LLM response in {llm_duration:.2f}s")
        
        # Validate response
        if not isinstance(response, dict) or "execution_plan" not in response:
            logger.warning(f"{STEP_NAME} | Invalid LLM response, using default plan")
            response = _build_default_plan(state, quick_category)
        
        # Ensure finish_tool is in plan
        plan_tools = [step.get("tool") for step in response.get("execution_plan", [])]
        if "finish_tool" not in plan_tools:
            response["execution_plan"].append({
                "step": len(response["execution_plan"]) + 1,
                "tool": "finish_tool",
                "reason": "Complete information gathering",
                "input_hint": None
            })
        
        duration = time.time() - start_time
        logger.info(f"{STEP_NAME} | ✅ Planning complete in {duration:.2f}s")
        logger.info(f"{STEP_NAME} | Complexity: {response.get('complexity', 'unknown')}")
        logger.info(f"{STEP_NAME} | Plan steps: {len(response.get('execution_plan', []))}")
        
        # Add metadata
        response["_planning_metadata"] = {
            "duration_seconds": duration,
            "llm_duration_seconds": llm_duration,
            "policy_section_used": policy_result.get("primary_section_name"),
            "quick_category": quick_category
        }
        
        return response
        
    except Exception as e:
        logger.error(f"{STEP_NAME} | ❌ Planning error: {e}", exc_info=True)
        
        # Return default plan on error
        default_plan = _build_default_plan(state, quick_category)
        default_plan["_planning_metadata"] = {
            "error": str(e),
            "fallback": True
        }
        
        return default_plan


def get_plan_context_for_agent(plan: Dict[str, Any], current_step: int = 0) -> str:
    """
    Convert execution plan to context string for ReACT agent.
    
    Args:
        plan: The execution plan from create_execution_plan()
        current_step: Current step index (0-based)
        
    Returns:
        Formatted string to include in agent context
    """
    if not plan or "execution_plan" not in plan:
        return ""
    
    lines = [
        "═══ EXECUTION PLAN ═══",
        f"Complexity: {plan.get('complexity', 'unknown')}",
        f"Customer Need: {plan.get('analysis', {}).get('customer_need', 'Unknown')}",
        ""
    ]
    
    # Policy info
    policy_info = plan.get("policy_applicable", {})
    if policy_info.get("policy_type") != "none":
        lines.append(f"📜 Applicable Policy: {policy_info.get('policy_type', 'N/A')}")
        reqs = policy_info.get("requirements_from_policy", [])
        if reqs:
            lines.append(f"   Required: {', '.join(reqs[:3])}")
        lines.append("")
    
    # Steps
    lines.append("Steps:")
    execution_plan = plan.get("execution_plan", [])
    for i, step in enumerate(execution_plan):
        step_num = step.get("step", i + 1)
        tool = step.get("tool", "unknown")
        reason = step.get("reason", "")[:50]
        
        if i < current_step:
            status = "✓"  # Completed
        elif i == current_step:
            status = "→"  # Current
        else:
            status = "○"  # Pending
        
        lines.append(f"  {status} Step {step_num}: {tool}")
        if reason:
            lines.append(f"      └─ {reason}")
    
    lines.append("")
    lines.append(f"Progress: {current_step}/{len(execution_plan)} steps")
    
    # Information needs summary
    info_needs = plan.get("information_needs", {})
    needed = [k.replace("_", " ") for k, v in info_needs.items() if v]
    if needed:
        lines.append(f"Looking for: {', '.join(needed[:4])}")
    
    lines.append("")
    lines.append("You may deviate from the plan if you discover new information.")
    lines.append("Always explain your reasoning when choosing a different tool.")
    
    return "\n".join(lines)


def should_follow_plan_step(
    plan: Dict[str, Any],
    current_step: int,
    gathered_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Determine if the agent should follow the next plan step or deviate.
    
    Args:
        plan: Execution plan
        current_step: Current step index
        gathered_info: What info has been gathered so far
        
    Returns:
        {
            "follow_plan": bool,
            "suggested_tool": str,
            "reason": str
        }
    """
    if not plan or "execution_plan" not in plan:
        return {"follow_plan": False, "suggested_tool": None, "reason": "No plan available"}
    
    execution_plan = plan.get("execution_plan", [])
    
    if current_step >= len(execution_plan):
        return {"follow_plan": False, "suggested_tool": "finish_tool", "reason": "Plan complete"}
    
    next_step = execution_plan[current_step]
    suggested_tool = next_step.get("tool")
    
    # Check if we should skip certain tools based on gathered info
    
    # Skip product_search if product already identified
    if suggested_tool == "product_search_tool" and gathered_info.get("product_identified"):
        return {
            "follow_plan": False,
            "suggested_tool": execution_plan[current_step + 1].get("tool") if current_step + 1 < len(execution_plan) else "finish_tool",
            "reason": "Product already identified, skipping product search"
        }
    
    # Skip attachment_analyzer if no attachments
    if suggested_tool == "attachment_analyzer_tool" and not gathered_info.get("has_attachments"):
        return {
            "follow_plan": False,
            "suggested_tool": execution_plan[current_step + 1].get("tool") if current_step + 1 < len(execution_plan) else "finish_tool",
            "reason": "No attachments to analyze"
        }
    
    # Skip vision_search if no images
    if suggested_tool in ["vision_search_tool", "ocr_image_analyzer_tool"] and not gathered_info.get("has_images"):
        return {
            "follow_plan": False,
            "suggested_tool": execution_plan[current_step + 1].get("tool") if current_step + 1 < len(execution_plan) else "finish_tool",
            "reason": "No images to analyze"
        }
    
    return {
        "follow_plan": True,
        "suggested_tool": suggested_tool,
        "reason": next_step.get("reason", "Following execution plan")
    }
