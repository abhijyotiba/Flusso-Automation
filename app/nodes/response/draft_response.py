"""
Draft Final Response Node
Generates customer-facing response based on all analysis.
"""

import logging
import time
import re
from typing import Dict, Any

from app.graph.state import TicketState
from app.utils.audit import add_audit_event
from app.clients.llm_client import call_llm
from app.config.constants import DRAFT_RESPONSE_PROMPT
from app.utils.detailed_logger import (
    log_node_start, log_node_complete, log_llm_interaction
)

logger = logging.getLogger(__name__)
STEP_NAME = "1️⃣4️⃣ DRAFT_RESPONSE"


def convert_to_html(text: str) -> str:
    """
    Convert markdown-style text to HTML for Freshdesk notes.
    Handles: bold, lists, paragraphs, headers, [VERIFY] tags
    
    Order of operations:
    1. Convert markdown to HTML tags FIRST
    2. Then escape remaining plain text content
    """
    # 1. Convert [VERIFY: ...] tags to highlighted spans FIRST (before escaping)
    text = re.sub(
        r'\[VERIFY:\s*([^\]]+)\]',
        r'|||VERIFY_START|||\1|||VERIFY_END|||',  # Temporary placeholder
        text
    )
    
    # 2. Convert **bold** to placeholder (before escaping)
    text = re.sub(r'\*\*([^*]+)\*\*', r'|||BOLD_START|||\1|||BOLD_END|||', text)
    
    # 3. Now escape HTML entities in plain text
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # 4. Replace placeholders with actual HTML tags
    text = text.replace('|||VERIFY_START|||', '<span style="background: #fef3c7; color: #92400e; padding: 2px 6px; border-radius: 4px; font-size: 12px;">⚠️ VERIFY: ')
    text = text.replace('|||VERIFY_END|||', '</span>')
    text = text.replace('|||BOLD_START|||', '<strong>')
    text = text.replace('|||BOLD_END|||', '</strong>')
    
    # Convert numbered lists (1. 2. 3.) and bullet lists
    lines = text.split('\n')
    in_numbered_list = False
    in_bullet_list = False
    result_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Check for numbered list item
        numbered_match = re.match(r'^(\d+)\.\s+(.+)$', stripped)
        if numbered_match:
            # Close bullet list if open
            if in_bullet_list:
                result_lines.append('</ul>')
                in_bullet_list = False
            # Open numbered list if not open
            if not in_numbered_list:
                result_lines.append('<ol style="margin: 12px 0; padding-left: 24px;">')
                in_numbered_list = True
            result_lines.append(f'<li style="margin: 8px 0;">{numbered_match.group(2)}</li>')
        # Check for bullet list item
        elif stripped.startswith('- ') or stripped.startswith('• '):
            # Close numbered list if open
            if in_numbered_list:
                result_lines.append('</ol>')
                in_numbered_list = False
            # Open bullet list if not open
            if not in_bullet_list:
                result_lines.append('<ul style="margin: 12px 0; padding-left: 24px;">')
                in_bullet_list = True
            result_lines.append(f'<li style="margin: 8px 0;">{stripped[2:]}</li>')
        else:
            # Close any open lists
            if in_numbered_list:
                result_lines.append('</ol>')
                in_numbered_list = False
            if in_bullet_list:
                result_lines.append('</ul>')
                in_bullet_list = False
            
            # Empty line = paragraph break
            if not stripped:
                result_lines.append('<br>')
            else:
                result_lines.append(f'<p style="margin: 8px 0; line-height: 1.6;">{stripped}</p>')
    
    # Close any open lists at end
    if in_numbered_list:
        result_lines.append('</ol>')
    if in_bullet_list:
        result_lines.append('</ul>')
    
    html = '\n'.join(result_lines)
    
    # Clean up multiple <br> tags
    html = re.sub(r'(<br>\s*){3,}', '<br><br>', html)
    
    # Wrap in a container div
    html = f'<div style="font-family: Arial, sans-serif; font-size: 14px; color: #1f2937; line-height: 1.6;">{html}</div>'
    
    return html


def draft_final_response(state: TicketState) -> Dict[str, Any]:
    """
    Generate final response to customer.

    Returns:
        Partial state update with:
            - draft_response
            - audit_events
    """
    start_time = time.time()
    logger.info(f"{STEP_NAME} | ▶ Generating customer response...")
    
    # Start node log
    node_log = log_node_start("draft_response", {})

    ticket_text = state.get("ticket_text", "") or ""
    subject = state.get("ticket_subject", "") or ""
    context = state.get("multimodal_context", "") or ""

    # Decision metrics
    enough_info = state.get("enough_information", False)
    risk = state.get("hallucination_risk", 1.0)
    confidence = state.get("product_match_confidence", 0.0)
    vip_ok = state.get("vip_compliant", True)
    
    # Vision quality metrics
    vision_quality = state.get("vision_match_quality", "NO_MATCH")
    vision_reason = state.get("vision_relevance_reason", "")
    vision_matched_cat = state.get("vision_matched_category", "")
    vision_expected_cat = state.get("vision_expected_category", "")
    
    node_log.input_summary = {
        "subject": subject[:100],
        "ticket_text_length": len(ticket_text),
        "context_length": len(context),
        "enough_info": enough_info,
        "hallucination_risk": risk,
        "product_confidence": confidence,
        "vip_compliant": vip_ok,
        "vision_quality": vision_quality
    }
    
    logger.info(f"{STEP_NAME} | 📥 Input: subject='{subject[:50]}...', context_len={len(context)}")
    logger.info(f"{STEP_NAME} | 📊 Metrics: enough_info={enough_info}, risk={risk:.2f}, confidence={confidence:.2f}, vip_ok={vip_ok}")
    logger.info(f"{STEP_NAME} | 🖼 Vision: quality={vision_quality}")

    # Build vision quality guidance for the LLM
    vision_guidance = ""
    if vision_quality == "CATEGORY_MISMATCH":
        vision_guidance = f"""
IMPORTANT - VISION MISMATCH DETECTED:
The customer asked about '{vision_expected_cat}' but our image search found '{vision_matched_cat}'.
DO NOT mention the mismatched products. Instead:
- Acknowledge you received the image
- Say you couldn't identify the specific product from the image
- Ask for model number, product code, or other identifying information
"""
    elif vision_quality == "NO_MATCH":
        vision_guidance = """
IMPORTANT - NO IMAGE MATCH:
Could not find a matching product from the attached image.
- Acknowledge you received the image
- Explain you couldn't find an exact match in the catalog
- Ask for additional details (model number, where purchased, etc.)
"""
    elif vision_quality == "LOW":
        vision_guidance = f"""
NOTE - UNCERTAIN IMAGE MATCH:
Visual matches have low confidence. Reason: {vision_reason}
- Present any product suggestions tentatively
- Ask customer to confirm if the suggested product is correct
"""

    meta = f"""
DECISION METRICS:
- Enough Information: {enough_info}
- Hallucination Risk: {risk:.2f}
- Product Confidence: {confidence:.2f}
- VIP Compliant: {vip_ok}
- Vision Match Quality: {vision_quality}
{vision_guidance}
"""

    user_prompt = f"""CUSTOMER TICKET:
Subject: {subject}
Description: {ticket_text}

RETRIEVED CONTEXT:
{context}

{meta}
"""

    try:
        logger.info(f"{STEP_NAME} | 🔄 Calling LLM to generate response...")
        llm_start = time.time()
        
        raw_response = call_llm(
            system_prompt=DRAFT_RESPONSE_PROMPT,
            user_prompt=user_prompt,
            response_format=None,  # plain text
        )
        
        llm_duration = time.time() - llm_start
        logger.info(f"{STEP_NAME} | ✓ LLM response in {llm_duration:.2f}s")

        # Ensure we end up with a string
        response_text = raw_response if isinstance(raw_response, str) else str(raw_response)
        
        # Calculate overall confidence score (0-100%)
        # Based on: product confidence (40%), hallucination risk inverted (40%), enough_info (20%)
        overall_confidence = (
            (confidence * 0.4) +           # Product match confidence
            ((1 - risk) * 0.4) +            # Inverted hallucination risk
            (0.2 if enough_info else 0)     # Enough information bonus
        ) * 100
        
        # Determine confidence label and color
        if overall_confidence >= 80:
            confidence_label = "🟢 HIGH"
            confidence_color = "#22c55e"  # green
        elif overall_confidence >= 50:
            confidence_label = "🟡 MEDIUM"
            confidence_color = "#eab308"  # yellow
        else:
            confidence_label = "🔴 LOW"
            confidence_color = "#ef4444"  # red
        
        # Convert markdown-style formatting in response to HTML
        html_response = convert_to_html(response_text)
        
        # Build HTML confidence header for Freshdesk
        confidence_header = f"""<div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); border-radius: 8px; padding: 16px; margin-bottom: 20px; font-family: Arial, sans-serif;">
    <div style="display: flex; align-items: center; margin-bottom: 12px;">
        <span style="font-size: 18px; margin-right: 8px;">📊</span>
        <span style="color: white; font-weight: bold; font-size: 16px;">AI CONFIDENCE:</span>
        <span style="background: {confidence_color}; color: white; padding: 4px 12px; border-radius: 12px; margin-left: 10px; font-weight: bold;">{confidence_label} ({overall_confidence:.0f}%)</span>
    </div>
    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
        <div style="background: rgba(255,255,255,0.1); padding: 8px 14px; border-radius: 6px;">
            <span style="color: #94a3b8; font-size: 12px;">Product Match</span><br>
            <span style="color: white; font-weight: bold; font-size: 16px;">{confidence*100:.0f}%</span>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 8px 14px; border-radius: 6px;">
            <span style="color: #94a3b8; font-size: 12px;">Info Quality</span><br>
            <span style="color: white; font-weight: bold; font-size: 16px;">{(1-risk)*100:.0f}%</span>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 8px 14px; border-radius: 6px;">
            <span style="color: #94a3b8; font-size: 12px;">Context</span><br>
            <span style="color: white; font-weight: bold; font-size: 16px;">{'✓ Available' if enough_info else '⚠ Limited'}</span>
        </div>
    </div>
</div>

"""
        response_with_confidence = confidence_header + html_response

        duration = time.time() - start_time
        logger.info(f"{STEP_NAME} | ✅ Generated response ({len(response_text)} chars) in {duration:.2f}s")
        logger.info(f"{STEP_NAME} | 📊 Overall confidence: {overall_confidence:.0f}% ({confidence_label})")
        logger.info(f"{STEP_NAME} | 📤 Preview: {response_text[:100]}..." if len(response_text) > 100 else f"{STEP_NAME} | 📤 Response: {response_text}")
        
        # Log LLM interaction
        log_llm_interaction(
            node_log,
            system_prompt=DRAFT_RESPONSE_PROMPT,
            user_prompt=user_prompt,
            response=response_text
        )
        log_node_complete(
            node_log,
            output_summary={
                "response_length": len(response_text),
                "overall_confidence": overall_confidence,
                "confidence_label": confidence_label,
                "duration_seconds": duration
            },
            llm_response=response_text
        )

        return {
            "draft_response": response_with_confidence,
            "overall_confidence": overall_confidence,
            "audit_events": add_audit_event(
                state,
                event="draft_final_response",
                event_type="GENERATION",
                details={
                    "response_length": len(response_text),
                    "llm_duration_seconds": llm_duration,
                    "overall_confidence": overall_confidence,
                    "confidence_label": confidence_label,
                },
            )["audit_events"],
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"{STEP_NAME} | ❌ Error after {duration:.2f}s: {e}", exc_info=True)
        fallback = (
            "We're reviewing your request and will get back to you shortly with a detailed response."
        )
        logger.warning(f"{STEP_NAME} | Using fallback response")

        return {
            "draft_response": fallback,
            "audit_events": add_audit_event(
                state,
                event="draft_final_response",
                event_type="ERROR",
                details={"error": str(e), "response_length": len(fallback)},
            )["audit_events"],
        }
