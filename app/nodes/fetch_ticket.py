"""
Fetch Ticket Node - FIXED VERSION
Now properly stores BOTH attachment metadata AND full attachment objects
"""

import logging
import time
from typing import Dict, Any

from app.graph.state import TicketState
from app.utils.audit import add_audit_event
from app.utils.attachment_processor import process_all_attachments
from app.clients.freshdesk_client import get_freshdesk_client
from app.utils.pii_masker import mask_email, mask_name
from app.utils.detailed_logger import (
    start_workflow_log, log_node_start, log_node_complete, get_current_log
)

logger = logging.getLogger(__name__)
STEP_NAME = "1️⃣ FETCH_TICKET"


def fetch_ticket_from_freshdesk(state: TicketState) -> Dict[str, Any]:
    start_time = time.time()
    
    raw_id = state.get("ticket_id")
    if not raw_id:
        raise ValueError("ticket_id is required")

    try:
        ticket_id = int(raw_id)
    except:
        raise ValueError(f"Invalid ticket_id: {raw_id}")

    logger.info(f"{'='*60}")
    logger.info(f"{STEP_NAME} | Starting for ticket #{ticket_id}")
    logger.info(f"{'='*60}")
    
    workflow_log = start_workflow_log(str(ticket_id))
    node_log = log_node_start("fetch_ticket", {"ticket_id": ticket_id})

    try:
        client = get_freshdesk_client()
        ticket = client.get_ticket(ticket_id, params={"include": "company,stats"})
        data = client.extract_ticket_data(ticket)

        # Check if already processed
        existing_tags = data.get("tags", [])
        ai_processed_tags = ["AI_PROCESSED", "AI_UNRESOLVED", "LOW_CONFIDENCE_MATCH", "VIP_RULE_FAILURE"]
        
        already_processed = any(tag in existing_tags for tag in ai_processed_tags)
        if already_processed:
            logger.warning(f"{STEP_NAME} | ⚠️ Ticket #{ticket_id} already has AI tags: {existing_tags}")
            return {
                "ticket_subject": data.get("subject", ""),
                "ticket_text": data.get("description", ""),
                "ticket_images": [],
                "requester_email": data.get("requester_email", ""),
                "requester_name": data.get("requester_name", "Unknown"),
                "tags": existing_tags,
                "has_text": False,
                "has_image": False,
                "ran_vision": True,
                "ran_text_rag": True,
                "ran_past_tickets": True,
                "should_skip": True,
                "skip_reason": f"Already processed (has tag: {[t for t in existing_tags if t in ai_processed_tags]})",
                "ticket_category": "already_processed",
                "audit_events": add_audit_event(
                    state,
                    event="fetch_ticket",
                    event_type="SKIP",
                    details={"ticket_id": ticket_id, "reason": "Already has AI processing tags", "existing_tags": existing_tags}
                )["audit_events"],
            }

        description = data.get("description", "")
        has_text = bool(description.strip())

        # ============================================================
        # CRITICAL FIX: Store BOTH processed content AND raw attachments
        # ============================================================
        raw_attachments = data.get("attachments", [])
        logger.info(f"{STEP_NAME} | 📎 Found {len(raw_attachments)} attachment(s)")
        
        # Process attachments for text extraction
        attachment_result = process_all_attachments(raw_attachments)
        
        images = attachment_result["images"]
        has_image = len(images) > 0
        attachment_text = attachment_result["extracted_content"]
        attachment_summary = attachment_result["attachment_summary"]
        attachment_stats = attachment_result["stats"]
        
        # Combine ticket description with attachment content
        if attachment_text:
            combined_text = f"{description}\n\n{'='*60}\n📎 ATTACHMENT CONTENT\n{'='*60}\n{attachment_text}"
            logger.info(f"{STEP_NAME} | 📄 Extracted {attachment_stats['total_chars']} chars from {attachment_stats['processed']} document(s)")
        else:
            combined_text = description

        # ============================================================
        # NEW: Store full attachment objects for tools
        # Filter to keep only document attachments (not images)
        # ============================================================
        document_attachments = []
        for att in raw_attachments:
            content_type = str(att.get("content_type", "")).lower()
            if not content_type.startswith("image/"):
                # Keep full attachment object with URL
                document_attachments.append({
                    "name": att.get("name", "unknown"),
                    "attachment_url": att.get("attachment_url"),
                    "content_type": content_type,
                    "size": att.get("size", 0)
                })
        
        logger.info(f"{STEP_NAME} | 📎 Prepared {len(document_attachments)} document attachment(s) for tools")

        updates = {
            "ticket_subject": data.get("subject", ""),
            "ticket_text": combined_text,
            "ticket_images": images,
            
            # Store BOTH for different purposes:
            "attachment_summary": attachment_summary,  # Metadata for display
            "ticket_attachments": document_attachments,  # Full objects for tools ✅ NEW
            
            "requester_email": data.get("requester_email", ""),
            "requester_name": data.get("requester_name", "Unknown"),
            "ticket_type": data.get("type"),
            "priority": data.get("priority"),
            "tags": data.get("tags", []),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "has_text": has_text or bool(attachment_text),
            "has_image": has_image,
            "ran_vision": False,
            "ran_text_rag": False,
            "ran_past_tickets": False,
        }

        updates["audit_events"] = add_audit_event(
            state,
            event="fetch_ticket",
            event_type="SUCCESS",
            details={
                "ticket_id": ticket_id,
                "has_text": updates["has_text"],
                "has_image": has_image,
                "image_count": len(images),
                "attachment_stats": attachment_stats,
                "document_attachments": len(document_attachments),  # ✅ NEW
                "tags": updates["tags"],
            }
        )["audit_events"]

        duration = time.time() - start_time
        logger.info(f"{STEP_NAME} | ✅ Completed in {duration:.2f}s")
        logger.info(f"{STEP_NAME} | Subject: {updates['ticket_subject'][:80]}...")
        logger.info(f"{STEP_NAME} | Text: {len(description)} chars | Images: {has_image} ({len(images)} files) | Docs: {len(document_attachments)}")
        logger.info(f"{STEP_NAME} | Requester: {mask_name(updates['requester_name'])} <{mask_email(updates['requester_email'])}>")
        
        if workflow_log:
            workflow_log.ticket_subject = updates['ticket_subject']
            workflow_log.ticket_text = combined_text
            workflow_log.ticket_images = images
            workflow_log.attachment_count = len(raw_attachments)
        
        log_node_complete(
            node_log,
            output_summary={
                "subject": updates['ticket_subject'][:100],
                "text_length": len(combined_text),
                "has_images": has_image,
                "image_count": len(images),
                "document_count": len(document_attachments),
                "attachment_stats": attachment_stats,
                "tags": updates['tags']
            }
        )

        return updates

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"{STEP_NAME} | ❌ FAILED after {duration:.2f}s: {e}", exc_info=True)

        return {
            "ticket_subject": "Error fetching ticket",
            "ticket_text": str(e),
            "ticket_images": [],
            "ticket_attachments": [],  # ✅ NEW - empty list on error
            "requester_email": "",
            "requester_name": "Unknown",
            "tags": [],
            "has_text": False,
            "has_image": False,
            "ran_vision": False,
            "ran_text_rag": False,
            "ran_past_tickets": False,
            "audit_events": add_audit_event(
                state,
                event="fetch_ticket",
                event_type="ERROR",
                details={"error": str(e)}
            )["audit_events"],
        }