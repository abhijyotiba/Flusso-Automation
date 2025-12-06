"""
Text RAG Pipeline Node
Retrieves relevant documents from Gemini File Search
"""

import logging
import time
from typing import Dict, Any

from app.graph.state import TicketState
from app.clients.gemini_client import get_gemini_client
from app.config.settings import settings
from app.utils.detailed_logger import (
    log_node_start, log_node_complete, log_text_rag_results
)

logger = logging.getLogger(__name__)
STEP_NAME = "4️⃣ TEXT_RAG"


def text_rag_pipeline(state: TicketState) -> Dict[str, Any]:
    """
    Retrieve relevant text documents using Gemini File Search
    
    Args:
        state: Current state
        
    Returns:
        Updated state with text_retrieval_results
    """
    start_time = time.time()
    logger.info(f"{STEP_NAME} | ▶ Starting text RAG pipeline")
    
    # Start node log
    node_log = log_node_start("text_rag_pipeline", {})
    
    text = state.get("ticket_text", "")
    subject = state.get("ticket_subject", "")
    
    node_log.input_summary = {
        "subject": subject[:100],
        "text_length": len(text),
        "text_preview": text[:500] if text else ""
    }
    
    logger.info(f"{STEP_NAME} | 📥 Input: subject='{subject[:50]}...', text_len={len(text)}")
    
    if not text.strip():
        duration = time.time() - start_time
        logger.info(f"{STEP_NAME} | ⏭ No text to process - skipping ({duration:.2f}s)")
        return {
            "text_retrieval_results": [],
            "ran_text_rag": True,
            "gemini_answer": "",
            "source_documents": [],
            "audit_events": state.get("audit_events", []) + [{
                "event": "text_rag_pipeline",
                "results_count": 0
            }]
        }
    
    logger.info(f"{STEP_NAME} | 🔄 Querying Gemini File Search...")
    
    try:
        client = get_gemini_client()
        top_k = settings.text_retrieval_top_k
        
        # Reformat ticket as a question to get better grounding from Gemini
        # Raw ticket text often doesn't return grounding sources
        query = f"""Based on the product documentation, help with this customer query:

Subject: {subject}

Customer Message:
{text}

Find relevant product information, part numbers, and documentation to address this query."""
        
        # Query file search with structured sources
        query_start = time.time()
        result = client.search_files_with_sources(query=query, top_k=top_k)
        query_duration = time.time() - query_start
        
        # Extract results
        hits = result.get('hits', [])
        gemini_answer = result.get('gemini_answer', '')
        source_documents = result.get('source_documents', [])
        
        duration = time.time() - start_time
        top_scores = [f"{h.get('score', 0):.3f}" for h in hits[:3]] if hits else []
        
        logger.info(f"{STEP_NAME} | ✅ Complete: {len(hits)} documents, {len(source_documents)} sources in {duration:.2f}s")
        logger.info(f"{STEP_NAME} | 📤 Top scores: {top_scores}")
        if gemini_answer:
            logger.info(f"{STEP_NAME} | 💬 Gemini answer: {gemini_answer[:100]}...")
        
        # Log detailed RAG results
        log_text_rag_results(node_log, hits, gemini_answer)
        log_node_complete(
            node_log,
            output_summary={
                "document_count": len(hits),
                "source_documents_count": len(source_documents),
                "top_scores": top_scores,
                "duration_seconds": duration
            },
            retrieval_results=[{
                "rank": i+1,
                "score": h.get("score", 0),
                "title": h.get("metadata", {}).get("title", "N/A"),
                "source": h.get("metadata", {}).get("source", "N/A"),
                "content_preview": h.get("content", "")[:500] if h.get("content") else ""
            } for i, h in enumerate(hits)],
            llm_response=gemini_answer
        )
        
        return {
            "text_retrieval_results": hits,
            "ran_text_rag": True,
            "gemini_answer": gemini_answer,
            "source_documents": source_documents,
            "audit_events": state.get("audit_events", []) + [{
                "event": "text_rag_pipeline",
                "results_count": len(hits),
                "source_documents_count": len(source_documents),
                "duration_seconds": duration
            }]
        }
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"{STEP_NAME} | ❌ Error after {duration:.2f}s: {e}", exc_info=True)
        return {
            "text_retrieval_results": [],
            "ran_text_rag": True,
            "gemini_answer": "",
            "source_documents": [],
            "audit_events": state.get("audit_events", []) + [{
                "event": "text_rag_pipeline",
                "error": str(e),
                "results_count": 0
            }]
        }
