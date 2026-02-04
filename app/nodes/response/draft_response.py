"""
Draft Final Response Node
Generates customer-facing response based on all analysis.
Enhanced with source citations for human agent review.
Enhanced with constraint validation for enforcement (Phase 3).
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
from app.services.policy_service import get_relevant_policy

# Constraint validator import (NEW)
try:
    from app.services.constraint_validator import (
        validate_constraints,
        format_constraints_for_prompt,
        post_validate_response,
        enforce_constraints_on_response,
        ConstraintResult,
    )
    CONSTRAINT_VALIDATOR_AVAILABLE = True
except ImportError:
    CONSTRAINT_VALIDATOR_AVAILABLE = False
    logging.getLogger(__name__).warning("Constraint validator not available in draft_response")

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


def build_collapsible_section(title: str, content: str, icon: str = "📋", default_open: bool = False) -> str:
    """
    Build a collapsible HTML section using <details> and <summary> tags.
    
    Args:
        title: The section title shown in the summary
        content: The HTML content to show when expanded
        icon: Emoji icon for the section
        default_open: Whether the section should be open by default
    
    Returns:
        HTML string with collapsible section
    """
    open_attr = "open" if default_open else ""
    return f"""
<details {open_attr} style="margin-bottom: 16px; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden;">
    <summary style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); padding: 12px 16px; cursor: pointer; font-weight: 600; color: #1e293b; display: flex; align-items: center; gap: 8px; user-select: none;">
        <span style="font-size: 16px;">{icon}</span>
        <span>{title}</span>
        <span style="margin-left: auto; font-size: 12px; color: #64748b;">Click to expand</span>
    </summary>
    <div style="padding: 16px; background: #ffffff;">
        {content}
    </div>
</details>
"""


def build_sources_html(
    source_documents: List[Dict[str, Any]],
    source_products: List[Dict[str, Any]],
    source_tickets: List[Dict[str, Any]],
    vision_quality: str = "LOW"
) -> str:
    """
    Build HTML section displaying all sources for the human agent.
    Each source type is in its own collapsible section for better readability.
    """
    sections = []
    
    # === RELEVANT DOCUMENTS (Collapsible) ===
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
        
        doc_table = f"""
            <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                <thead>
                    <tr style="background: #f3f4f6;">
                        <th style="padding: 8px; text-align: left; width: 40px;">#</th>
                        <th style="padding: 8px; text-align: left;">Document</th>
                        <th style="padding: 8px; text-align: center; width: 80px;">Relevance</th>
                    </tr>
                </thead>
                <tbody>{doc_rows}</tbody>
            </table>"""
        
        sections.append(build_collapsible_section(
            title=f"Relevant Documents ({len(source_documents[:5])} found)",
            content=doc_table,
            icon="📄",
            default_open=False
        ))
    
    # === VISUAL MATCHES (Collapsible) ===
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
        
        product_table = f"""
            <div style="margin-bottom: 8px;">{quality_badge}</div>
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
            </table>"""
        
        sections.append(build_collapsible_section(
            title=f"Visual Matches ({len(source_products[:5])} found)",
            content=product_table,
            icon="🖼️",
            default_open=False
        ))
    elif vision_quality == "CATEGORY_MISMATCH":
        mismatch_content = '<p style="margin: 0; font-size: 13px; color: #991b1b;">Visual search found products from a different category than what the customer is asking about. These results have been excluded.</p>'
        sections.append(build_collapsible_section(
            title="Visual Matches ❌ CATEGORY MISMATCH",
            content=mismatch_content,
            icon="🖼️",
            default_open=False
        ))
    
    # === PAST TICKETS (Collapsible) ===
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
        
        ticket_table = f"""
            <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                <thead>
                    <tr style="background: #f3f4f6;">
                        <th style="padding: 8px; text-align: left; width: 80px;">Ticket</th>
                        <th style="padding: 8px; text-align: left;">Subject</th>
                        <th style="padding: 8px; text-align: center; width: 70px;">Similarity</th>
                    </tr>
                </thead>
                <tbody>{ticket_rows}</tbody>
            </table>"""
        
        sections.append(build_collapsible_section(
            title=f"Similar Past Tickets ({len(source_tickets[:5])} found)",
            content=ticket_table,
            icon="🎫",
            default_open=False
        ))
    
    if not sections:
        return """
        <div style="background: #fef3c7; border: 1px solid #fcd34d; border-radius: 8px; padding: 12px; margin-top: 20px;">
            <p style="margin: 0; color: #92400e; font-size: 13px;">⚠️ No source documents, product matches, or similar tickets were found for this request.</p>
        </div>"""
    
    # Wrap all sections in a collapsible sources container
    sources_content = ''.join(sections)
    sources_html = f"""
    <div style="margin-top: 24px; padding-top: 20px; border-top: 2px solid #e5e7eb;">
        <details style="border: 1px solid #cbd5e1; border-radius: 8px; overflow: hidden;">
            <summary style="background: linear-gradient(135deg, #374151 0%, #4b5563 100%); padding: 14px 18px; cursor: pointer; font-weight: bold; color: white; display: flex; align-items: center; gap: 10px; user-select: none;">
                <span style="font-size: 18px;">📎</span>
                <span style="font-size: 15px;">SOURCES & REFERENCES</span>
                <span style="margin-left: auto; font-size: 12px; color: #d1d5db; font-weight: normal;">▼ Click to view AI source data</span>
            </summary>
            <div style="padding: 16px; background: #fafafa;">
                {sources_content}
            </div>
        </details>
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
    
    # Extract customer message - handle both dict and string formats
    if isinstance(info_request_response, dict):
        customer_message_check = info_request_response.get("customer_message", "")
    else:
        customer_message_check = str(info_request_response) if info_request_response else ""
    
    # Only treat as legitimate info request if:
    # 1. needs_more_info flag is set
    # 2. There's actual content in info_request_response
    # 3. Evidence decision is request_info or escalate
    # 4. Confidence is actually low
    # 5. The customer_message is not empty (empty = use LLM-generated response)
    is_legitimate_info_request = (
        needs_more_info 
        and info_request_response 
        and customer_message_check.strip()  # Must have actual message content
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
    
    # === Missing Requirements Check (for returns/replacements/warranty) ===
    # IMPORTANT: Agent's missing_requirements from finish_tool takes precedence
    # Agent may flag "clearer photo needed" even when image exists but doesn't show defect
    agent_missing_requirements = state.get("missing_requirements", []) or []
    ticket_images = state.get("ticket_images", []) or []
    ticket_attachments = state.get("ticket_attachments", []) or []
    
    # === Image Analysis Insights (from OCR analyzer) ===
    # This tells us if the image shows a defect or if product appears in good condition
    image_analysis_insights = state.get("image_analysis_insights", []) or []
    image_shows_defect = False
    image_condition_note = ""
    
    if image_analysis_insights:
        for insight in image_analysis_insights:
            condition = insight.get("condition", "").lower()
            image_type = insight.get("image_type", "")
            description = insight.get("description", "")
            
            # Check if image analysis found a defect/damage
            if image_type == "damaged_item" or any(word in condition for word in ["damage", "defect", "broken", "crack", "leak", "worn"]):
                image_shows_defect = True
            # Check if image shows product in good/excellent condition (no defect visible)
            elif any(word in condition for word in ["excellent", "good", "new", "mint"]) and not image_shows_defect:
                image_condition_note = f"Image received shows product in '{condition[:50]}' - defect not clearly visible"
        
        logger.info(f"{STEP_NAME} | 🖼️ Image analysis: shows_defect={image_shows_defect}, condition_note='{image_condition_note[:60]}...'")
    
    # Initialize missing_requirements with agent's assessment
    missing_requirements = list(agent_missing_requirements) if agent_missing_requirements else []
    
    # Auto-detect missing requirements for return/replacement/warranty tickets
    # BUT respect agent's assessment - if agent says "clearer photo needed", don't override
    if ticket_category in ["return_refund", "replacement_parts", "warranty_claim", "product_issue"]:
        ticket_lower = (ticket_text + " " + subject).lower()
        
        # Check for PO/order number
        has_po = any(keyword in ticket_lower for keyword in [
            "po ", "po:", "po#", "purchase order", "order #", "order number", 
            "invoice", "receipt", "confirmation"
        ])
        
        # Check for shipping address
        has_address = any(keyword in ticket_lower for keyword in [
            "address", "ship to", "send to", "deliver to", "street", "city", "zip", "state"
        ])
        
        # Check for photo/video evidence
        has_media = len(ticket_images) > 0 or any(
            att.get("content_type", "").startswith("image") or 
            att.get("content_type", "").startswith("video")
            for att in ticket_attachments
        )
        
        # Build auto-detected missing requirements list
        auto_missing = []
        if not has_po and not any("po" in req.lower() or "purchase" in req.lower() for req in agent_missing_requirements):
            auto_missing.append("PO/Purchase Order number or proof of purchase")
        
        # For photo requirement: 
        # - If no image at all, ask for photo
        # - If image exists but agent flagged "clearer photo needed", respect that
        # - If image exists and shows defect, don't ask for photo
        # - If image exists but doesn't show defect clearly (condition=excellent), ask for clearer photo
        photo_requirement_from_agent = any("photo" in req.lower() or "video" in req.lower() or "image" in req.lower() for req in agent_missing_requirements)
        
        if ticket_category in ["warranty_claim", "product_issue", "replacement_parts"]:
            if photo_requirement_from_agent:
                # Agent already flagged this - don't add duplicate
                pass
            elif not has_media:
                # No image at all
                auto_missing.append("Photo or video showing the issue/defect")
            elif has_media and image_condition_note and not image_shows_defect:
                # Image exists but doesn't show defect clearly
                auto_missing.append("Clearer photo or video showing the specific defect/issue area")
        
        if not has_address and ticket_category in ["return_refund", "replacement_parts", "warranty_claim"]:
            if not any("address" in req.lower() for req in agent_missing_requirements):
                auto_missing.append("Shipping address for replacement delivery")
        
        # Combine: agent's requirements take priority, then add auto-detected
        missing_requirements = list(set(agent_missing_requirements + auto_missing))
        
        logger.info(f"{STEP_NAME} | 🔍 Requirements check: PO={has_po}, Media={has_media}, Address={has_address}")
        logger.info(f"{STEP_NAME} | 📋 Agent flagged: {agent_missing_requirements}")
        logger.info(f"{STEP_NAME} | ⚠️ Final missing requirements: {missing_requirements}")

    # === Fetch Policy Context for Response Generation ===
    policy_context = ""
    policy_requirements_list = []
    try:
        policy_result = get_relevant_policy(
            ticket_category=ticket_category,
            ticket_text=ticket_text,
            keywords=None
        )
        policy_context = policy_result.get("primary_section", "")[:2500]  # Limit for prompt size
        policy_requirements_list = policy_result.get("policy_requirements", [])
        policy_section_name = policy_result.get("primary_section_name", "General")
        logger.info(f"{STEP_NAME} | 📜 Policy loaded: {policy_section_name} ({len(policy_context)} chars, {len(policy_requirements_list)} requirements)")
    except Exception as e:
        logger.warning(f"{STEP_NAME} | ⚠️ Could not load policy: {e}")
        policy_context = ""
        policy_requirements_list = []

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
    missing_requirements_guidance = ""
    
    # Build missing requirements guidance if applicable
    if missing_requirements:
        missing_list = "\n".join(f"   - {req}" for req in missing_requirements)
        missing_requirements_guidance = f"""
═══════════════════════════════════════════════════════════════════════
🔴 MISSING REQUIREMENTS - MUST REQUEST BEFORE PROCEEDING
═══════════════════════════════════════════════════════════════════════
The following required information is MISSING from the customer's ticket:
{missing_list}

⚠️ DO NOT approve or promise to process this request until ALL above items are received.
⚠️ Your response MUST politely ask the customer to provide the missing information.

Example response format:
"We're happy to help with your [request type]! To process this, we need a few more details:
{missing_list}

Once we receive this information, we can proceed with your request."
═══════════════════════════════════════════════════════════════════════
"""
    
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
    elif ticket_category == "return_refund":
        category_guidance = f"""
⚠️ CATEGORY: RETURN/REFUND REQUEST
- Customer wants to return a product or get a refund
- MANDATORY: Collect PO/order number, reason for return, and photo if defective
- Reference return policy (45/90/180 day windows with restocking fees)
- State RGA number will be issued after verification
- DO NOT approve the return without required information
{missing_requirements_guidance}
"""
    elif ticket_category in ["replacement_parts", "warranty_claim", "product_issue"]:
        category_guidance = f"""
⚠️ CATEGORY: {ticket_category.upper().replace('_', ' ')}
- Customer is reporting a product issue or requesting replacement
- MANDATORY: Before processing, verify we have:
  1. PO/Purchase Order or proof of purchase
  2. Photo/video showing the issue/defect  
  3. Shipping address for replacement
- DO NOT approve replacement without required information
- If information is missing, politely request it
{missing_requirements_guidance}
"""
    elif ticket_category in ["shipping_tracking"]:
        category_guidance = f"""
⚠️ CATEGORY: {ticket_category.upper().replace('_', ' ')}
- This is about order logistics, not product identification
- Focus on the customer's order/tracking request
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

    # Build image analysis guidance if we have insights about image condition
    image_analysis_guidance = ""
    if image_condition_note:
        image_analysis_guidance = f"""
═══════════════════════════════════════════════════════════════════════
📸 IMAGE ANALYSIS RESULT - IMPORTANT
═══════════════════════════════════════════════════════════════════════
We DID receive and analyze the customer's attached image.
Result: {image_condition_note}

⚠️ CRITICAL: DO NOT say "the photo did not come through" or "we didn't receive the image"
The image WAS received and processed - it just doesn't clearly show the defect.

CORRECT response: "We received your photo. However, we couldn't clearly identify the 
discoloration/defect in the image. Could you please send a closer photo specifically 
highlighting the affected area?"

INCORRECT response: "Please re-send the photo as it did not come through on our end."
═══════════════════════════════════════════════════════════════════════
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
{image_analysis_guidance}
"""

    # Build policy section for prompt
    policy_prompt_section = ""
    if policy_context or policy_requirements_list:
        requirements_text = "\n".join(f"  - {req}" for req in policy_requirements_list) if policy_requirements_list else "  (See policy text below)"
        policy_prompt_section = f"""
═══════════════════════════════════════════════════════════════════════
📜 COMPANY POLICY - YOU MUST FOLLOW THESE RULES
═══════════════════════════════════════════════════════════════════════

**Key Requirements for this request type:**
{requirements_text}

**Full Policy Section:**
{policy_context}

⚠️ IMPORTANT: Your response MUST comply with the policy above. 
If the policy requires specific information (PO, photos, address, etc.) 
that the customer has NOT provided, you MUST ask for it before proceeding.
═══════════════════════════════════════════════════════════════════════
"""

    # === CONSTRAINT VALIDATION (Phase 3) ===
    # Get or compute constraints from ticket_facts and category
    constraint_result = state.get("constraint_result")
    constraints_prompt_section = state.get("constraints_prompt_section", "")
    
    if CONSTRAINT_VALIDATOR_AVAILABLE and not constraint_result:
        # Compute constraints if not already in state
        ticket_facts = state.get("ticket_facts", {})
        product_text = state.get("product_text", "") or ""
        try:
            raw_constraint_result = validate_constraints(
                ticket_facts=ticket_facts,
                ticket_category=ticket_category,
                product_text=product_text
            )
            # Convert to dict for state storage and later use
            constraint_result = raw_constraint_result.to_dict()
            constraints_prompt_section = format_constraints_for_prompt(constraint_result)
            logger.info(f"{STEP_NAME} | 🔒 Computed constraints: missing_fields={len(constraint_result.get('missing_fields', []))}, required_citations={len(constraint_result.get('required_citations', []))}")
        except Exception as e:
            logger.warning(f"{STEP_NAME} | ⚠️ Constraint validation failed: {e}")
            constraint_result = None
            constraints_prompt_section = ""
    elif constraint_result and not constraints_prompt_section:
        # Format if we have result but not formatted prompt
        try:
            constraints_prompt_section = format_constraints_for_prompt(constraint_result)
        except Exception as e:
            logger.warning(f"{STEP_NAME} | ⚠️ Constraint formatting failed: {e}")
            constraints_prompt_section = ""

    user_prompt = f"""CUSTOMER TICKET:
Subject: {subject}
Description: {ticket_text}

TICKET CATEGORY: {ticket_category}
{policy_prompt_section}
{constraints_prompt_section}
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
        
        # === POST-VALIDATION: Ensure constraints are enforced in response ===
        constraint_validation_result = None
        constraint_warnings = []
        
        if CONSTRAINT_VALIDATOR_AVAILABLE and constraint_result:
            try:
                # Validate that LLM response includes required asks and citations
                constraint_validation_result = post_validate_response(
                    response_text=response_text,
                    constraints=constraint_result
                )
                
                # Extract warnings from validation
                if not constraint_validation_result.get("valid", True):
                    constraint_warnings = constraint_validation_result.get("warnings", [])
                    logger.warning(f"{STEP_NAME} | ⚠️ Constraint validation warnings: {constraint_warnings}")
                    
                    # Attempt to auto-fix response if critical items missing
                    if constraint_warnings:
                        try:
                            enforced_response = enforce_constraints_on_response(
                                response_text=response_text,
                                constraints=constraint_result
                            )
                            if enforced_response != response_text:
                                logger.info(f"{STEP_NAME} | 🔧 Auto-enforced missing citations/asks in response")
                                response_text = enforced_response
                                html_response = convert_to_html(response_text)
                        except Exception as enforce_err:
                            logger.warning(f"{STEP_NAME} | ⚠️ Auto-enforce failed: {enforce_err}")
                else:
                    logger.info(f"{STEP_NAME} | ✅ Response passed constraint validation")
                    
            except Exception as validation_err:
                logger.warning(f"{STEP_NAME} | ⚠️ Constraint post-validation failed: {validation_err}")
        
        # === Build sources section ===
        sources_html = build_sources_html(
            source_documents=source_documents,
            source_products=source_products,
            source_tickets=source_tickets,
            vision_quality=vision_quality
        )
        
        # Build compact confidence banner (always visible at top)
        confidence_banner = f"""<div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; font-family: Arial, sans-serif;">
    <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 12px;">
        <span style="font-size: 16px;">🤖</span>
        <span style="color: white; font-weight: bold; font-size: 15px;">AI Draft Response</span>
        <span style="background: {confidence_color}; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 13px;">{confidence_label} ({overall_confidence:.0f}%)</span>
        <span style="color: #94a3b8; font-size: 12px; margin-left: auto;">Product: {confidence*100:.0f}% | Info: {(1-risk)*100:.0f}%</span>
    </div>
</div>
"""
        
        # Wrap the AI response in a highlighted section (open by default)
        response_section = f"""
<details open style="margin-bottom: 16px; border: 2px solid #3b82f6; border-radius: 8px; overflow: hidden;">
    <summary style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); padding: 12px 16px; cursor: pointer; font-weight: 600; color: #1e40af; display: flex; align-items: center; gap: 8px; user-select: none;">
        <span style="font-size: 16px;">💬</span>
        <span>SUGGESTED RESPONSE TO CUSTOMER</span>
        <span style="margin-left: auto; font-size: 11px; color: #64748b; font-weight: normal;">Copy this to reply</span>
    </summary>
    <div style="padding: 16px; background: #ffffff; border-top: 1px solid #bfdbfe;">
        {html_response}
    </div>
</details>
"""

        # Build collapsible AI analysis details section
        analysis_details = f"""
<details style="margin-bottom: 16px; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden;">
    <summary style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); padding: 12px 16px; cursor: pointer; font-weight: 600; color: #334155; display: flex; align-items: center; gap: 8px; user-select: none;">
        <span style="font-size: 16px;">📊</span>
        <span>AI Analysis Details</span>
        <span style="margin-left: auto; font-size: 11px; color: #64748b; font-weight: normal;">▼ Click to expand</span>
    </summary>
    <div style="padding: 16px; background: #ffffff;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px;">
            <div style="background: #f8fafc; padding: 12px; border-radius: 6px; text-align: center;">
                <div style="color: #64748b; font-size: 11px; text-transform: uppercase;">Product Match</div>
                <div style="color: #1e293b; font-weight: bold; font-size: 20px; margin-top: 4px;">{confidence*100:.0f}%</div>
            </div>
            <div style="background: #f8fafc; padding: 12px; border-radius: 6px; text-align: center;">
                <div style="color: #64748b; font-size: 11px; text-transform: uppercase;">Info Quality</div>
                <div style="color: #1e293b; font-weight: bold; font-size: 20px; margin-top: 4px;">{(1-risk)*100:.0f}%</div>
            </div>
            <div style="background: #f8fafc; padding: 12px; border-radius: 6px; text-align: center;">
                <div style="color: #64748b; font-size: 11px; text-transform: uppercase;">Context</div>
                <div style="color: #1e293b; font-weight: bold; font-size: 20px; margin-top: 4px;">{'✓' if enough_info else '⚠'}</div>
            </div>
            <div style="background: #f8fafc; padding: 12px; border-radius: 6px; text-align: center;">
                <div style="color: #64748b; font-size: 11px; text-transform: uppercase;">VIP Check</div>
                <div style="color: #1e293b; font-weight: bold; font-size: 20px; margin-top: 4px;">{'✓' if vip_ok else '❌'}</div>
            </div>
        </div>
        <div style="margin-top: 12px; padding: 10px; background: #f1f5f9; border-radius: 6px; font-size: 12px; color: #475569;">
            <strong>Category:</strong> {ticket_category} | <strong>Vision Quality:</strong> {vision_quality}
        </div>
    </div>
</details>
"""
        # === Build resource links section (only if product identified with high confidence) ===
        resource_links_html = ""
        try:
            resource_links_html = get_resource_links_for_response(
                identified_product=identified_product,
                product_confidence=product_confidence_for_links
            )
            if resource_links_html:
                # Wrap resource links in collapsible section
                resource_links_html = build_collapsible_section(
                    title="Product Resources & Links",
                    content=resource_links_html,
                    icon="🔗",
                    default_open=False
                )
                logger.info(f"{STEP_NAME} | 📎 Added resource links for identified product")
        except Exception as e:
            logger.warning(f"{STEP_NAME} | ⚠️ Failed to get resource links: {e}")
            resource_links_html = ""
        
        # Combine: banner + response section + analysis details + resource links + sources + agent console
        # Structure:
        # 1. Compact banner (always visible) - shows confidence at a glance
        # 2. Response section (open by default) - the actual response to copy
        # 3. Analysis details (collapsed) - detailed metrics
        # 4. Resource links (collapsed) - product links
        # 5. Sources (collapsed) - documents, products, tickets
        # 6. Agent console button
        response_with_confidence = confidence_banner + response_section + analysis_details + resource_links_html + sources_html + build_agent_console_section()

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
                "identified_product_model": identified_product.get("model") if identified_product else None,
                "constraint_validation_passed": constraint_validation_result.get("valid", True) if constraint_validation_result else None,
                "constraint_warnings_count": len(constraint_warnings) if constraint_warnings else 0,
            },
            llm_response=response_text
        )

        return {
            "draft_response": response_with_confidence,
            "overall_confidence": overall_confidence,
            "constraint_validation_result": constraint_validation_result,
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
                    "constraint_validation_passed": constraint_validation_result.get("valid", True) if constraint_validation_result else None,
                    "constraint_warnings": constraint_warnings if constraint_warnings else None,
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
