"""
HTML Formatters for Draft Response

This module contains utility functions for generating formatted HTML
content for Freshdesk notes and agent panels.

Functions:
- convert_to_html: Convert markdown-style text to HTML
- build_collapsible_section: Create collapsible HTML sections
- build_sources_html: Build HTML for source citations
- build_agent_console_section: Build agent console button section
"""

import re
from typing import Dict, Any, List


def convert_to_html(text: str) -> str:
    """
    Convert markdown-style text to HTML for Freshdesk notes.
    Handles: bold, lists, paragraphs, headers, [VERIFY] tags
    
    Order of operations:
    1. Convert markdown to HTML tags FIRST
    2. Then escape remaining plain text content
    
    Args:
        text: Markdown-style text to convert
        
    Returns:
        HTML formatted string
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
    
    Args:
        source_documents: List of retrieved document sources
        source_products: List of matched product sources
        source_tickets: List of similar past tickets
        vision_quality: Quality level of vision match ("HIGH", "LOW", "NO_MATCH", "CATEGORY_MISMATCH")
        
    Returns:
        HTML string with all source sections
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
    <div style="margin-top: 16px;">
        <details style="margin-bottom: 16px; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden;">
            <summary style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); padding: 12px 16px; cursor: pointer; font-weight: 600; color: #1e293b; display: flex; align-items: center; gap: 8px; user-select: none;">
                <span style="font-size: 16px;">📎</span>
                <span>Sources & References</span>
                <span style="margin-left: auto; font-size: 12px; color: #64748b;">Click to expand</span>
            </summary>
            <div style="padding: 16px; background: #ffffff;">
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
    
    Returns:
        HTML string with agent console button
    """
    from app.config.settings import settings
    url = settings.agent_console_url
    return f"""
    <div style="margin-top:16px; padding-top:12px; border-top:1px dashed #e5e7eb; display:flex; align-items:center; gap:12px;">
        <a href="{url}" target="_blank" rel="noopener noreferrer" style="display:inline-block; background:#0ea5e9; color:#ffffff; padding:10px 14px; border-radius:8px; text-decoration:none; font-weight:600;">Open Agent Console</a>
        <div style="color:#475569; font-size:13px;">Agent Console: lookup product details by <strong>model no.</strong> or <strong>product ID</strong>.</div>
    </div>
    """
