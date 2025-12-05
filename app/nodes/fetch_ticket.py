"""
Fetch Ticket Node
Clean + Correct Production Version
Now with comprehensive attachment processing (PDFs, DOCX, XLSX, TXT)
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

# Step counter for workflow tracing
STEP_NAME = "1️⃣ FETCH_TICKET"


def fetch_ticket_from_freshdesk(state: TicketState) -> Dict[str, Any]:
    start_time = time.time()
    
    # -------------------------------------------------
    # Validate ticket_id
    # -------------------------------------------------
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
    
    # Start detailed workflow log
    workflow_log = start_workflow_log(str(ticket_id))
    node_log = log_node_start("fetch_ticket", {"ticket_id": ticket_id})

    try:
        client = get_freshdesk_client()

        # Fetch ticket with company + stats (requester auto-included)
        ticket = client.get_ticket(
            ticket_id,
            params={"include": "company,stats"}
        )

        data = client.extract_ticket_data(ticket)

        # -------------------------------------------------
        # CHECK IF ALREADY PROCESSED BY AI
        # Skip tickets that already have AI processing tags
        # -------------------------------------------------
        existing_tags = data.get("tags", [])
        ai_processed_tags = ["AI_PROCESSED", "AI_UNRESOLVED", "LOW_CONFIDENCE_MATCH", "VIP_RULE_FAILURE"]
        
        already_processed = any(tag in existing_tags for tag in ai_processed_tags)
        if already_processed:
            logger.warning(f"{STEP_NAME} | ⚠️ Ticket #{ticket_id} already has AI tags: {existing_tags}")
            logger.warning(f"{STEP_NAME} | Marking for skip to prevent duplicate processing")
            # Return minimal state with skip flag
            return {
                "ticket_subject": data.get("subject", ""),
                "ticket_text": data.get("description", ""),
                "ticket_images": [],
                "requester_email": data.get("requester_email", ""),
                "requester_name": data.get("requester_name", "Unknown"),
                "tags": existing_tags,
                "has_text": False,
                "has_image": False,
                "ran_vision": True,  # Mark as done to skip RAG
                "ran_text_rag": True,
                "ran_past_tickets": True,
                "should_skip": True,
                "skip_reason": f"Already processed (has tag: {[t for t in existing_tags if t in ai_processed_tags]})",
                "skip_private_note": "",  # No note needed, already processed
                "ticket_category": "already_processed",
                "audit_events": add_audit_event(
                    state,
                    event="fetch_ticket",
                    event_type="SKIP",
                    details={
                        "ticket_id": ticket_id,
                        "reason": "Already has AI processing tags",
                        "existing_tags": existing_tags
                    }
                )["audit_events"],
            }

        # -------------------------------------------------
        # Extract text description
        # -------------------------------------------------
        description = data.get("description", "")
        has_text = bool(description.strip())

        # -------------------------------------------------
        # Process ALL attachments (PDFs, DOCX, XLSX, TXT + Images)
        # -------------------------------------------------
        attachments = data.get("attachments", [])
        logger.info(f"{STEP_NAME} | 📎 Found {len(attachments)} attachment(s)")
        
        attachment_result = process_all_attachments(attachments)
        
        # Images for vision pipeline
        images = attachment_result["images"]
        has_image = len(images) > 0
        
        # Extracted text from documents (PDFs, DOCX, etc.)
        attachment_text = attachment_result["extracted_content"]
        attachment_summary = attachment_result["attachment_summary"]
        attachment_stats = attachment_result["stats"]
        
        # Combine ticket description with attachment content
        if attachment_text:
            combined_text = f"{description}\n\n{'='*60}\n📎 ATTACHMENT CONTENT\n{'='*60}\n{attachment_text}"
            logger.info(f"{STEP_NAME} | 📄 Extracted {attachment_stats['total_chars']} chars from {attachment_stats['processed']} document(s)")
        else:
            combined_text = description

        # -------------------------------------------------
        # Build state update
        # -------------------------------------------------
        updates = {
            "ticket_subject": data.get("subject", ""),
            "ticket_text": combined_text,  # Now includes attachment content!
            "ticket_images": images,
            
            # Store attachment metadata
            "attachment_summary": attachment_summary,

            "requester_email": data.get("requester_email", ""),
            "requester_name": data.get("requester_name", "Unknown"),

            "ticket_type": data.get("type"),
            "priority": data.get("priority"),
            "tags": data.get("tags", []),

            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),

            "has_text": has_text or bool(attachment_text),
            "has_image": has_image,

            # Required for RAG loop
            "ran_vision": False,
            "ran_text_rag": False,
            "ran_past_tickets": False,
        }

        # -------------------------------------------------
        # Audit
        # -------------------------------------------------
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
                "tags": updates["tags"],
            }
        )["audit_events"]

        duration = time.time() - start_time
        logger.info(f"{STEP_NAME} | ✅ Completed in {duration:.2f}s")
        logger.info(f"{STEP_NAME} | Subject: {updates['ticket_subject'][:80]}...")
        logger.info(f"{STEP_NAME} | Text length: {len(description)} chars | Has images: {has_image} ({len(images)} files)")
        # Log with PII masking for privacy compliance
        logger.info(f"{STEP_NAME} | Requester: {mask_name(updates['requester_name'])} <{mask_email(updates['requester_email'])}>")
        logger.info(f"{STEP_NAME} | Tags: {updates['tags']}")
        
        # Update workflow log with ticket info
        if workflow_log:
            workflow_log.ticket_subject = updates['ticket_subject']
            workflow_log.ticket_text = combined_text
            workflow_log.ticket_images = images
            workflow_log.attachment_count = len(attachments)
        
        # Complete node log
        log_node_complete(
            node_log,
            output_summary={
                "subject": updates['ticket_subject'][:100],
                "text_length": len(combined_text),
                "has_images": has_image,
                "image_count": len(images),
                "attachment_stats": attachment_stats,
                "tags": updates['tags']
            }
        )

        return updates

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"{STEP_NAME} | ❌ FAILED after {duration:.2f}s: {e}", exc_info=True)

        # Safe fallback
        return {
            "ticket_subject": "Error fetching ticket",
            "ticket_text": str(e),
            "ticket_images": [],
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
