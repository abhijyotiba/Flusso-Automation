"""
Draft Final Response Node
Generates customer-facing response based on all analysis.
Enhanced with source citations for human agent review.
"""

import logging
import time
import re
from datetime import datetime
from typing import Dict, Any, List

from app.graph.state import TicketState
from app.utils.audit import add_audit_event
from app.clients.llm_client import call_llm
from app.config.constants import DRAFT_RESPONSE_PROMPT, ENHANCED_DRAFT_RESPONSE_PROMPT
from app.utils.detailed_logger import (
    log_node_start, log_node_complete, log_llm_interaction
)
from app.services.resource_links_service import get_resource_links_for_response

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


def build_sources_html(
    source_documents: List[Dict[str, Any]],
    source_products: List[Dict[str, Any]],
    source_tickets: List[Dict[str, Any]],
    vision_quality: str = "LOW"
) -> str:
    """
    Build HTML section displaying all sources for the human agent.
    """
    sections = []
    
    # === RELEVANT DOCUMENTS ===
    if source_documents:
        doc_rows = ""
        for doc in source_documents[:5]:  # Limit to 5
            title = doc.get('title', 'Unknown Document')[:50]
            score = doc.get('relevance_score', 0)
            stars = "⭐⭐⭐" if score >= 0.85 else "⭐⭐" if score >= 0.7 else "⭐"
            doc_rows += f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #e5e7eb;">{doc.get('rank', '-')}</td>
                <td style="padding: 8px; border-bottom: 1px solid #e5e7eb;">{title}</td>
                <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; text-align: center;">{stars}</td>
            </tr>"""
        
        sections.append(f"""
        <div style="margin-bottom: 20px;">
            <h4 style="color: #1e40af; margin-bottom: 10px;">📄 Relevant Documents</h4>
            <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                <thead>
                    <tr style="background: #f3f4f6;">
                        <th style="padding: 8px; text-align: left; width: 40px;">#</th>
                        <th style="padding: 8px; text-align: left;">Document</th>
                        <th style="padding: 8px; text-align: center; width: 80px;">Relevance</th>
                    </tr>
                </thead>
                <tbody>{doc_rows}</tbody>
            </table>
        </div>""")
    
    # === VISUAL MATCHES ===
    if source_products and vision_quality != "CATEGORY_MISMATCH":
        product_rows = ""
        for prod in source_products[:5]:  # Limit to 5
            title = prod.get('product_title', 'Unknown')[:40]
            model = prod.get('model_no', 'N/A')
            score = prod.get('similarity_score', 0)
            match_icon = prod.get('match_level', '🟡')
            product_rows += f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #e5e7eb;">{match_icon}</td>
                <td style="padding: 8px; border-bottom: 1px solid #e5e7eb;">{title}</td>
                <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; font-family: monospace;">{model}</td>
                <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; text-align: center;">{score}%</td>
            </tr>"""
        
        quality_badge = {
            "HIGH": '<span style="background: #22c55e; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px;">HIGH CONFIDENCE</span>',
            "LOW": '<span style="background: #eab308; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px;">LOW CONFIDENCE</span>',
            "NO_MATCH": '<span style="background: #6b7280; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px;">NO MATCH</span>',
        }.get(vision_quality, '')
        
        sections.append(f"""
        <div style="margin-bottom: 20px;">
            <h4 style="color: #1e40af; margin-bottom: 10px;">🖼️ Visual Matches {quality_badge}</h4>
            <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                <thead>
                    <tr style="background: #f3f4f6;">
                        <th style="padding: 8px; text-align: left; width: 30px;"></th>
                        <th style="padding: 8px; text-align: left;">Product</th>
                        <th style="padding: 8px; text-align: left; width: 120px;">Model</th>
                        <th style="padding: 8px; text-align: center; width: 60px;">Match</th>
                    </tr>
                </thead>
                <tbody>{product_rows}</tbody>
            </table>
        </div>""")
    elif vision_quality == "CATEGORY_MISMATCH":
        sections.append(f"""
        <div style="margin-bottom: 20px; background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 12px;">
            <h4 style="color: #dc2626; margin-bottom: 8px;">🖼️ Visual Matches ❌ CATEGORY MISMATCH</h4>
            <p style="margin: 0; font-size: 13px; color: #991b1b;">Visual search found products from a different category than what the customer is asking about. These results have been excluded.</p>
        </div>""")
    
    # === PAST TICKETS ===
    if source_tickets:
        ticket_rows = ""
        for ticket in source_tickets[:5]:  # Limit to 5
            if not ticket or not isinstance(ticket, dict):
                continue
            ticket_id = ticket.get('ticket_id', 'N/A')
            subject_raw = ticket.get('subject', 'Unknown') or 'Unknown'
            subject = str(subject_raw)[:45]
            score = ticket.get('similarity_score', 0)
            ticket_rows += f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; font-family: monospace; color: #6366f1;">#{ticket_id}</td>
                <td style="padding: 8px; border-bottom: 1px solid #e5e7eb;">{subject}</td>
                <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; text-align: center;">{score}%</td>
            </tr>"""
        
        sections.append(f"""
        <div style="margin-bottom: 20px;">
            <h4 style="color: #1e40af; margin-bottom: 10px;">🎫 Similar Past Tickets</h4>
            <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                <thead>
                    <tr style="background: #f3f4f6;">
                        <th style="padding: 8px; text-align: left; width: 80px;">Ticket</th>
                        <th style="padding: 8px; text-align: left;">Subject</th>
                        <th style="padding: 8px; text-align: center; width: 70px;">Similarity</th>
                    </tr>
                </thead>
                <tbody>{ticket_rows}</tbody>
            </table>
        </div>""")
    
    if not sections:
        return """
        <div style="background: #fef3c7; border: 1px solid #fcd34d; border-radius: 8px; padding: 12px; margin-top: 20px;">
            <p style="margin: 0; color: #92400e; font-size: 13px;">⚠️ No source documents, product matches, or similar tickets were found for this request.</p>
        </div>"""
    
    # Wrap all sections in a sources container
    sources_html = f"""
    <div style="margin-top: 24px; padding-top: 20px; border-top: 2px solid #e5e7eb;">
        <h3 style="color: #374151; font-size: 16px; margin-bottom: 16px;">📎 SOURCES</h3>
        {''.join(sections)}
    </div>"""
    
    return sources_html


def build_agent_console_section() -> str:
    """
    Small HTML section with a button linking to the Agent Console.
    Appears at the bottom of all draft responses to help human agents
    quickly lookup product details by model number or product ID.
    """
    from app.config.settings import settings
    url = settings.agent_console_url
    return f"""
    <div style="margin-top:16px; padding-top:12px; border-top:1px dashed #e5e7eb; display:flex; align-items:center; gap:12px;">
        <a href="{url}" target="_blank" rel="noopener noreferrer" style="display:inline-block; background:#0ea5e9; color:#ffffff; padding:10px 14px; border-radius:8px; text-decoration:none; font-weight:600;">Open Agent Console</a>
        <div style="color:#475569; font-size:13px;">Agent Console: lookup product details by <strong>model no.</strong> or <strong>product ID</strong>.</div>
    </div>
    """


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
    
    # === Check for SYSTEM ERRORS first (API failures, timeouts, etc.) ===
    # These should NOT send "need more info" to customers - they're internal errors
    workflow_error = state.get("workflow_error")
    workflow_error_type = state.get("workflow_error_type")
    is_system_error = state.get("is_system_error", False)
    
    if workflow_error or is_system_error:
        logger.error(f"{STEP_NAME} | ❌ SYSTEM ERROR detected: {workflow_error}")
        
        ticket_subject = state.get("ticket_subject", "No subject")
        ticket_text = state.get("ticket_text", "")[:500]
        requester_name = state.get("requester_name", "Customer")
        error_node = state.get("workflow_error_node", "Unknown")
        
        # Create internal error note (NOT a customer message)
        error_header = f"""<div style="background: linear-gradient(135deg, #475569 0%, #64748b 100%); border-radius: 8px; padding: 16px; margin-bottom: 20px; font-family: Arial, sans-serif;">
    <div style="display: flex; align-items: center; margin-bottom: 12px;">
        <span style="font-size: 18px; margin-right: 8px;">⚙️</span>
        <span style="color: white; font-weight: bold; font-size: 16px;">PROCESSING INTERRUPTED - MANUAL REVIEW</span>
    </div>
    <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 6px;">
        <span style="color: #e2e8f0; font-size: 13px;">The AI assistant couldn't complete processing. Please review and respond to this ticket.</span>
    </div>
</div>
"""
        
        # Build detailed error note for agent
        error_details = f"""
<div style="padding: 16px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 16px;">
    <h4 style="color: #334155; margin-bottom: 12px;">📋 TICKET SUMMARY</h4>
    <p style="margin: 4px 0;"><strong>Subject:</strong> {ticket_subject}</p>
    <p style="margin: 4px 0;"><strong>Customer:</strong> {requester_name}</p>
    <p style="margin: 4px 0; font-size: 13px;"><strong>Preview:</strong> {ticket_text[:300]}...</p>
</div>

<div style="padding: 16px; background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 8px;">
    <h4 style="color: #475569; margin-bottom: 8px;">ℹ️ TECHNICAL DETAILS</h4>
    <p style="margin: 4px 0;"><strong>Issue:</strong> {workflow_error_type or 'Processing interrupted'}</p>
    <p style="margin: 4px 0;"><strong>Stage:</strong> {error_node}</p>
    <p style="margin: 4px 0; font-size: 12px; color: #64748b;"><strong>Details:</strong> {workflow_error or 'No additional details'}</p>
</div>

<div style="padding: 16px; background: #f0f9ff; border: 1px solid #7dd3fc; border-radius: 8px; margin-top: 16px;">
    <h4 style="color: #0369a1; margin-bottom: 8px;">📝 NEXT STEPS</h4>
    <p style="margin: 4px 0; font-size: 13px;">Please review the ticket and respond to the customer manually. This may have been a temporary service issue.</p>
</div>
"""
        
        duration = time.time() - start_time
        logger.info(f"{STEP_NAME} | Generated system error note in {duration:.2f}s")
        
        return {
            "draft_response": error_header + error_details + build_agent_console_section(),
            "overall_confidence": 0.0,
            "is_system_error": True,
            "audit_events": add_audit_event(
                state,
                event="draft_final_response",
                event_type="SYSTEM_ERROR",
                details={
                    "error_type": workflow_error_type,
                    "error_message": workflow_error,
                    "error_node": error_node,
                },
            )["audit_events"],
        }

    # === Check for evidence-based info request (LEGITIMATE need for more info) ===
    # Only show this when evidence analysis ACTUALLY determined we need more info
    info_request_response = state.get("info_request_response", "")
    needs_more_info = state.get("needs_more_info", False)
    evidence_analysis = state.get("evidence_analysis", {})
    
    # Make sure this is a legitimate info request, not a fallback
    evidence_decision = evidence_analysis.get("resolution_action", "")
    is_legitimate_info_request = (
        needs_more_info 
        and info_request_response 
        and evidence_decision in ["request_info", "escalate"]
        and evidence_analysis.get("final_confidence", 0) < 0.5  # Only if actually low confidence
    )
    
    if is_legitimate_info_request:
        logger.info(f"{STEP_NAME} | ℹ️ Legitimate info request - evidence confidence: {evidence_analysis.get('final_confidence', 0):.0%}")
        
        # Handle both dict and string formats for info_request_response
        if isinstance(info_request_response, dict):
            customer_message = info_request_response.get("customer_message", "")
            private_note = info_request_response.get("private_note", "")
        else:
            # Legacy format: string response
            customer_message = str(info_request_response)
            private_note = evidence_analysis.get("private_note", "")
        
        # Convert to HTML
        html_response = convert_to_html(customer_message)
        
        # Build simple info request header
        info_header = f"""<div style="background: #0369a1; border-radius: 6px; padding: 12px; margin-bottom: 16px;">
    <span style="color: white; font-weight: bold;">ℹ️ Information Needed</span>
</div>

"""
        # Add private note section for human agent (compact)
        private_note_html = ""
        if private_note:
            private_note_html = f"""
<div style="margin-top: 16px; padding: 12px; background: #f8fafc; border-left: 3px solid #0369a1; font-size: 12px;">
    <strong>🔒 Agent Note:</strong><br/>
    <span style="color: #475569;">{convert_to_html(private_note)}</span>
</div>
"""
        
        response_with_header = info_header + html_response + private_note_html + build_agent_console_section()
        
        duration = time.time() - start_time
        logger.info(f"{STEP_NAME} | ✅ Generated info-request response in {duration:.2f}s")
        
        log_node_complete(
            node_log,
            output_summary={
                "response_type": "info_request",
                "evidence_decision": evidence_analysis.get("resolution_action", "UNKNOWN"),
                "evidence_confidence": evidence_analysis.get("final_confidence", 0),
                "response_length": len(customer_message),
                "duration_seconds": duration,
            },
            llm_response=customer_message
        )
        
        return {
            "draft_response": response_with_header,
            "overall_confidence": 0.0,  # Low confidence since we need more info
            "audit_events": add_audit_event(
                state,
                event="draft_final_response",
                event_type="INFO_REQUEST",
                details={
                    "response_type": "legitimate_info_request",
                    "evidence_decision": evidence_analysis.get("resolution_action", "UNKNOWN"),
                    "evidence_confidence": evidence_analysis.get("final_confidence", 0),
                    "response_length": len(customer_message),
                    "has_private_note": bool(private_note),
                },
            )["audit_events"],
        }

    ticket_text = state.get("ticket_text", "") or ""
    subject = state.get("ticket_subject", "") or ""
    context = state.get("multimodal_context", "") or ""
    ticket_category = state.get("ticket_category", "general") or "general"

    # Get evidence resolver analysis for confidence metrics
    evidence_analysis = state.get("evidence_analysis", {})
    evidence_confidence = evidence_analysis.get("final_confidence", 0.0) if evidence_analysis else 0.0
    
    # Decision metrics - use evidence_resolver's confidence as primary source
    enough_info = state.get("enough_information", False)
    confidence = state.get("product_match_confidence", evidence_confidence)
    # Derive risk from evidence confidence (low confidence = higher risk)
    risk = max(0.0, 1.0 - confidence) if confidence > 0 else state.get("hallucination_risk", 0.3)
    vip_ok = state.get("vip_compliant", True)
    
    # Vision quality metrics
    vision_quality = state.get("vision_match_quality", "NO_MATCH")
    vision_reason = state.get("vision_relevance_reason", "")
    vision_matched_cat = state.get("vision_matched_category", "")
    vision_expected_cat = state.get("vision_expected_category", "")
    
    # === Source data for citations ===
    source_documents = state.get("source_documents", []) or []
    source_products = state.get("source_products", []) or []
    source_tickets = state.get("source_tickets", []) or []
    gemini_answer = state.get("gemini_answer", "") or ""
    
    # === Identified product for resource links ===
    identified_product = state.get("identified_product", None)
    product_confidence_for_links = state.get("product_confidence", 0.0) or confidence
    
    node_log.input_summary = {
        "subject": subject[:100],
        "ticket_text_length": len(ticket_text),
        "context_length": len(context),
        "ticket_category": ticket_category,
        "enough_info": enough_info,
        "evidence_confidence": evidence_confidence,
        "product_confidence": confidence,
        "vip_compliant": vip_ok,
        "vision_quality": vision_quality,
        "source_documents_count": len(source_documents),
        "source_products_count": len(source_products),
        "source_tickets_count": len(source_tickets)
    }
    
    logger.info(f"{STEP_NAME} | 📥 Input: subject='{subject[:50]}...', category={ticket_category}, context_len={len(context)}")
    logger.info(f"{STEP_NAME} | 📊 Metrics: enough_info={enough_info}, confidence={confidence:.2f}, evidence_conf={evidence_confidence:.2f}, vip_ok={vip_ok}")
    logger.info(f"{STEP_NAME} | 🖼 Vision: quality={vision_quality}")
    logger.info(f"{STEP_NAME} | 📎 Sources: docs={len(source_documents)}, products={len(source_products)}, tickets={len(source_tickets)}")

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

    # Get current date for accurate date comparisons
    current_date = datetime.now().strftime("%m-%d-%Y")  # e.g., "12-05-2025" (MM-DD-YYYY)
    current_date_readable = datetime.now().strftime("%B %d, %Y")  # e.g., "December 05, 2025"

    # Build category-specific guidance
    category_guidance = ""
    if ticket_category == "pricing_request":
        category_guidance = """
⚠️ CATEGORY: PRICING REQUEST
- Customer is asking for pricing/MSRP
- DO NOT ask for product photos or receipts
- Provide pricing if found in search results
- If pricing not found, say we will follow up with pricing information
"""
    elif ticket_category == "dealer_inquiry":
        category_guidance = """
⚠️ CATEGORY: DEALER/PARTNERSHIP INQUIRY  
- Customer wants to become a dealer or partner
- Acknowledge their interest in Flusso partnership
- If they submitted documents (application, resale certificate), acknowledge receipt
- Provide next steps for application review
- DO NOT ask for product photos or model numbers
"""
    elif ticket_category in ["shipping_tracking", "return_refund"]:
        category_guidance = f"""
⚠️ CATEGORY: {ticket_category.upper().replace('_', ' ')}
- This is about order logistics, not product identification
- Focus on the customer's order/return request
- DO NOT ask for product photos unless needed for the return
"""
    elif ticket_category == "general":
        category_guidance = """
⚠️ CATEGORY: GENERAL INQUIRY / ACCOUNT UPDATE
- This is a general request that does NOT involve product support
- Could be: account update, address change, business name change, contact info update, or general question
- DO NOT ask for product model numbers or photos
- DO NOT assume this is a product-related request
- Acknowledge the customer's request/information professionally
- If it's an account/address update: Confirm you've received the information and will update records
- If it's a general question: Answer directly or acknowledge you'll look into it
- Keep response brief and professional
"""

    meta = f"""
TODAY'S DATE: {current_date} ({current_date_readable})
Use this date for ALL date comparisons. Any date before {current_date} is in the PAST.
Date format is MM-DD-YYYY.

TICKET CATEGORY: {ticket_category}
{category_guidance}

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

TICKET CATEGORY: {ticket_category}

RETRIEVED CONTEXT:
{context}

{meta}
"""

    try:
        logger.info(f"{STEP_NAME} | 🔄 Calling LLM to generate enhanced response...")
        llm_start = time.time()
        
        # Use enhanced prompt for structured response
        raw_response = call_llm(
            system_prompt=ENHANCED_DRAFT_RESPONSE_PROMPT,
            user_prompt=user_prompt,
            response_format=None,  # plain text
        )
        
        llm_duration = time.time() - llm_start
        logger.info(f"{STEP_NAME} | ✓ LLM response in {llm_duration:.2f}s")

        # Ensure we end up with a valid string
        response_text = ""
        if raw_response is not None:
            response_text = str(raw_response).strip()
        
        # Check if response is actually valid (not empty, not "None", minimum length)
        if not response_text or response_text.lower() == "none" or len(response_text) < 20:
            logger.warning(f"{STEP_NAME} | ⚠ LLM returned invalid response: '{response_text}' - using fallback")
            response_text = (
                f"Thank you for reaching out about: {subject}\n\n"
                "We have received your request and our team is reviewing the details. "
                "We will get back to you shortly with a detailed response.\n\n"
                "If you have any additional information to share, please reply to this ticket."
            )
        
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
            confidence_label = "⚪ LOW"
            confidence_color = "#6b7280"  # red
        
        # Convert markdown-style formatting in response to HTML
        html_response = convert_to_html(response_text)
        
        # === Build sources section ===
        sources_html = build_sources_html(
            source_documents=source_documents,
            source_products=source_products,
            source_tickets=source_tickets,
            vision_quality=vision_quality
        )
        
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
        # === Build resource links section (only if product identified with high confidence) ===
        resource_links_html = ""
        try:
            resource_links_html = get_resource_links_for_response(
                identified_product=identified_product,
                product_confidence=product_confidence_for_links
            )
            if resource_links_html:
                logger.info(f"{STEP_NAME} | 📎 Added resource links for identified product")
        except Exception as e:
            logger.warning(f"{STEP_NAME} | ⚠️ Failed to get resource links: {e}")
            resource_links_html = ""
        
        # Combine: confidence header + response + resource links + sources + agent console
        # Resource links appear below suggested response, above sources
        response_with_confidence = confidence_header + html_response + resource_links_html + sources_html + build_agent_console_section()

        duration = time.time() - start_time
        logger.info(f"{STEP_NAME} | ✅ Generated response ({len(response_text)} chars) in {duration:.2f}s")
        logger.info(f"{STEP_NAME} | 📊 Overall confidence: {overall_confidence:.0f}% ({confidence_label})")
        logger.info(f"{STEP_NAME} | 📤 Preview: {response_text[:100]}..." if len(response_text) > 100 else f"{STEP_NAME} | 📤 Response: {response_text}")
        
        # Log LLM interaction
        log_llm_interaction(
            node_log,
            system_prompt=ENHANCED_DRAFT_RESPONSE_PROMPT,
            user_prompt=user_prompt,
            response=response_text
        )
        log_node_complete(
            node_log,
            output_summary={
                "response_length": len(response_text),
                "overall_confidence": overall_confidence,
                "confidence_label": confidence_label,
                "duration_seconds": duration,
                "source_documents_count": len(source_documents),
                "source_products_count": len(source_products),
                "source_tickets_count": len(source_tickets),
                "resource_links_added": bool(resource_links_html),
                "identified_product_model": identified_product.get("model") if identified_product else None
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
                    "source_documents_count": len(source_documents),
                    "source_products_count": len(source_products),
                    "source_tickets_count": len(source_tickets),
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
