"""
Past Tickets RAG Node
Queries Pinecone resolved tickets DB
"""

import logging
import time
from typing import Dict, Any, List

from app.graph.state import TicketState
from app.utils.audit import add_audit_event
from app.clients.embeddings import embed_text
from app.clients.pinecone_client import get_pinecone_client
from app.config.settings import settings
from app.utils.detailed_logger import (
    log_node_start, log_node_complete, log_past_tickets_results
)

logger = logging.getLogger(__name__)
STEP_NAME = "5️⃣ PAST_TICKETS"


def retrieve_past_tickets(state: TicketState) -> Dict[str, Any]:
    start_time = time.time()
    logger.info(f"{STEP_NAME} | ▶ Starting past tickets search")
    
    # Start node log
    node_log = log_node_start("past_tickets", {})
    
    text = state.get("ticket_text", "")
    node_log.input_summary = {"text_length": len(text)}
    logger.info(f"{STEP_NAME} | 📥 Input: text_len={len(text)}")

    if not text.strip():
        duration = time.time() - start_time
        logger.info(f"{STEP_NAME} | ⏭ No text to search - skipping ({duration:.2f}s)")
        return {
            "past_ticket_results": [],
            "ran_past_tickets": True,
            "audit_events": add_audit_event(
                state,
                "retrieve_past_tickets",
                "EMPTY",
                {"results_count": 0},
            )["audit_events"],
        }

    logger.info(f"{STEP_NAME} | 🔄 Embedding query + searching Pinecone...")

    try:
        client = get_pinecone_client()
        
        embed_start = time.time()
        vector = embed_text(text)
        embed_duration = time.time() - embed_start
        logger.info(f"{STEP_NAME} | ✓ Text embedded in {embed_duration:.2f}s, dim={len(vector) if vector else 0}")
        
        top_k = settings.past_ticket_top_k
        
        query_start = time.time()
        hits = client.query_past_tickets(vector, top_k=top_k)
        query_duration = time.time() - query_start

        duration = time.time() - start_time
        top_scores = [f"{h.get('score', 0):.3f}" for h in hits[:3]] if hits else []
        
        logger.info(f"{STEP_NAME} | ✅ Complete: {len(hits)} similar tickets in {duration:.2f}s (query: {query_duration:.2f}s)")
        logger.info(f"{STEP_NAME} | 📤 Top scores: {top_scores}")
        
        # Log detailed past tickets results
        log_past_tickets_results(node_log, hits)
        log_node_complete(
            node_log,
            output_summary={
                "results_count": len(hits),
                "top_scores": top_scores,
                "duration_seconds": duration
            },
            retrieval_results=[{
                "rank": i+1,
                "score": h.get("score", 0),
                "ticket_id": h.get("metadata", {}).get("ticket_id", "N/A"),
                "subject": h.get("metadata", {}).get("subject", "N/A"),
                "resolution": h.get("metadata", {}).get("resolution", "N/A")
            } for i, h in enumerate(hits)]
        )
        
        # === Build structured source_tickets for citations ===
        source_tickets = []
        for i, hit in enumerate(hits[:5]):  # Limit to top 5 for display
            meta = hit.get("metadata", {}) or {}
            score = hit.get("score", 0)
            
            source_tickets.append({
                "rank": i + 1,
                "ticket_id": meta.get("ticket_id", "N/A"),
                "subject": meta.get("subject", "Unknown Subject"),
                "resolution_type": meta.get("resolution_type", "N/A"),
                "resolution_summary": meta.get("resolution", meta.get("resolution_summary", ""))[:200],
                "similarity_score": round(score * 100),  # As percentage
                "source_type": "past_tickets"
            })
        
        logger.info(f"{STEP_NAME} | 🎫 Created {len(source_tickets)} structured ticket sources")

        return {
            "past_ticket_results": hits,
            "source_tickets": source_tickets,
            "ran_past_tickets": True,
            "audit_events": add_audit_event(
                state,
                "retrieve_past_tickets",
                "SEARCH",
                {"results_count": len(hits), "source_tickets_count": len(source_tickets), "duration_seconds": duration},
            )["audit_events"],
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"{STEP_NAME} | ❌ Error after {duration:.2f}s: {e}", exc_info=True)

        return {
            "past_ticket_results": [],
            "source_tickets": [],
            "ran_past_tickets": True,
            "audit_events": add_audit_event(
                state,
                "retrieve_past_tickets",
                "ERROR",
                {"error": str(e)},
            )["audit_events"]
        }
