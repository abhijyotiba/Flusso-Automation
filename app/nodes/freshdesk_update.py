"""
Freshdesk Update Node
Updates ticket with public/private reply + tags
"""

import logging
import time
from typing import Dict, Any

from app.graph.state import TicketState
from app.utils.audit import add_audit_event
from app.clients.freshdesk_client import get_freshdesk_client
from app.config.constants import ResolutionStatus

logger = logging.getLogger(__name__)
STEP_NAME = "1️⃣6️⃣ FRESHDESK_UPDATE"


def _handle_skipped_ticket(state: TicketState, ticket_id: int, start_time: float) -> Dict[str, Any]:
    """
    Handle tickets that skipped the full workflow (PO, auto-reply, spam, already_processed).
    Only adds private note + tags, no public response.
    """
    category = state.get("ticket_category", "unknown")
    skip_reason = state.get("skip_reason", "Unknown")
    private_note = state.get("private_note", "")
    suggested_tags = state.get("suggested_tags", [])
    
    logger.info(f"{STEP_NAME} | 🚀 SKIP MODE: category={category}")
    logger.info(f"{STEP_NAME} | Skip reason: {skip_reason}")
    
    # If already processed by AI, do nothing (don't add notes or update tags)
    if category == "already_processed":
        duration = time.time() - start_time
        logger.info(f"{STEP_NAME} | ⏭️ Already processed - no Freshdesk update needed")
        return {
            "tags": state.get("tags", []),
            "resolution_status": "already_processed",
            "audit_events": add_audit_event(
                state,
                "update_freshdesk_ticket",
                "SKIP",
                {
                    "ticket_id": ticket_id,
                    "category": category,
                    "reason": "Already processed by AI - no update performed",
                    "duration_seconds": duration,
                },
            )["audit_events"],
        }
    
    client = get_freshdesk_client()
    
    try:
        # Add private note explaining why skipped
        if private_note:
            logger.info(f"{STEP_NAME} | 📝 Adding skip private note")
            note_start = time.time()
            client.add_note(ticket_id, private_note, private=True)
            logger.info(f"{STEP_NAME} | ✓ Private note added in {time.time() - note_start:.2f}s")
        
        # Update tags (merge with existing)
        old_tags = state.get("tags") or []
        merged_tags = sorted(list(set(old_tags + suggested_tags)))
        
        # Only update if there are new tags to add
        if suggested_tags:
            logger.info(f"{STEP_NAME} | 🏷 Updating tags: {old_tags} + {suggested_tags} → {merged_tags}")
            tags_start = time.time()
            client.update_ticket(ticket_id, tags=merged_tags)
            logger.info(f"{STEP_NAME} | ✓ Tags updated in {time.time() - tags_start:.2f}s")
        else:
            logger.info(f"{STEP_NAME} | 🏷 No new tags to add, skipping tag update")
        
        duration = time.time() - start_time
        logger.info(f"{STEP_NAME} | ✅ SKIPPED: ticket #{ticket_id} updated (no public response) in {duration:.2f}s")
        
        return {
            "tags": merged_tags,
            "resolution_status": "skipped",
            "audit_events": add_audit_event(
                state,
                "update_freshdesk_ticket",
                "SKIP",
                {
                    "ticket_id": ticket_id,
                    "category": category,
                    "skip_reason": skip_reason,
                    "note_type": "private_only",
                    "tags": merged_tags,
                    "duration_seconds": duration,
                },
            )["audit_events"],
        }
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"{STEP_NAME} | ❌ Error updating skipped ticket #{ticket_id}: {e}", exc_info=True)
        
        return {
            "audit_events": add_audit_event(
                state,
                "update_freshdesk_ticket",
                "ERROR",
                {"ticket_id": ticket_id, "error": str(e), "skip_mode": True},
            )["audit_events"]
        }


def update_freshdesk_ticket(state: TicketState) -> Dict[str, Any]:
    """Final step → push replies + tags to Freshdesk."""
    start_time = time.time()
    logger.info(f"{STEP_NAME} | ▶ Starting Freshdesk update...")

    try:
        ticket_id = int(state.get("ticket_id"))
    except Exception:
        raise ValueError("Invalid ticket_id in state")

    # Check if this is a skipped ticket (PO, auto-reply, spam)
    skip_workflow_applied = state.get("skip_workflow_applied", False)
    
    if skip_workflow_applied:
        return _handle_skipped_ticket(state, ticket_id, start_time)

    status = state.get("resolution_status", ResolutionStatus.AI_UNRESOLVED.value)
    reply_text = state.get("final_response_public") or ""

    logger.info(f"{STEP_NAME} | 📥 Input: ticket_id={ticket_id}, status='{status}', reply_len={len(reply_text)}")

    client = get_freshdesk_client()

    try:
        # ---------------- ALL AI RESPONSES ARE PRIVATE NOTES ----------------
        # Human agents review and send public responses manually
        unresolved = status in [
            ResolutionStatus.AI_UNRESOLVED.value,
            ResolutionStatus.LOW_CONFIDENCE_MATCH.value,
            ResolutionStatus.VIP_RULE_FAILURE.value,
        ]

        if unresolved:
            # Private note with review needed flag
            note_text = f"""
🤖 <b>AI Review Needed</b>

<b>Status:</b> {status}

<b>Suggested Reply:</b>
{reply_text}

<b>Decision Metrics:</b>
• Product Confidence: {state.get('product_match_confidence', 0):.2f}
• Hallucination Risk: {state.get('hallucination_risk', 0):.2f}
• Enough Info: {state.get('enough_information', False)}
• VIP Compliant: {state.get('vip_compliant', True)}
"""
            logger.info(f"{STEP_NAME} | 📝 Adding PRIVATE note (needs human review)")
        else:
            # Private note with draft response for agent to review and send
            note_text = reply_text
            logger.info(f"{STEP_NAME} | 📝 Adding PRIVATE note (AI draft for agent review)")

        note_start = time.time()
        client.add_note(ticket_id, note_text, private=True)
        logger.info(f"{STEP_NAME} | ✓ Private note added in {time.time() - note_start:.2f}s")
        note_type = "private"

        # ---------------------- UPDATE TAGS ----------------------
        old_tags = state.get("tags") or []
        extra_tags = state.get("extra_tags") or []
        merged_tags = sorted(list(set(old_tags + extra_tags)))

        logger.info(f"{STEP_NAME} | 🏷 Updating tags: {old_tags} + {extra_tags} → {merged_tags}")
        tags_start = time.time()
        client.update_ticket(ticket_id, tags=merged_tags)
        logger.info(f"{STEP_NAME} | ✓ Tags updated in {time.time() - tags_start:.2f}s")

        duration = time.time() - start_time
        logger.info(f"{STEP_NAME} | ✅ Complete: ticket #{ticket_id} updated ({note_type} note) in {duration:.2f}s")

        return {
            "tags": merged_tags,
            "audit_events": add_audit_event(
                state,
                "update_freshdesk_ticket",
                "UPDATE",
                {
                    "ticket_id": ticket_id,
                    "resolution_status": status,
                    "note_type": note_type,
                    "tags": merged_tags,
                    "duration_seconds": duration,
                },
            )["audit_events"],
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"{STEP_NAME} | ❌ Error updating ticket #{ticket_id} after {duration:.2f}s: {e}", exc_info=True)

        return {
            "audit_events": add_audit_event(
                state,
                "update_freshdesk_ticket",
                "ERROR",
                {"ticket_id": ticket_id, "error": str(e)},
            )["audit_events"]
        }
