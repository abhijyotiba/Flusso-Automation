"""
Detailed Logger Utility
Captures comprehensive node responses and stores them in structured JSON logs.

This logger captures:
- Full responses from all nodes (vision, RAG, LLM decisions)
- Retrieval results with scores and metadata
- LLM prompts and responses
- Timing information
- Error details

Logs are stored in: workflow_logs/ticket_{id}_{timestamp}.json

Thread-safe implementation using thread-local storage for concurrent webhooks.
"""

import logging
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# Thread-safe storage for concurrent workflow logs
_workflow_logs: Dict[int, "WorkflowLog"] = {}
_logs_lock = threading.Lock()

LOG_DIR = Path("workflow_logs")


@dataclass
class NodeExecution:
    """Represents a single node execution with all details"""
    node_name: str
    start_time: float
    end_time: float = 0.0
    duration_seconds: float = 0.0
    status: str = "started"
    
    # Input/Output
    input_summary: Dict[str, Any] = field(default_factory=dict)
    output_summary: Dict[str, Any] = field(default_factory=dict)
    
    # For retrieval nodes
    retrieval_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # For LLM nodes
    llm_prompt: Optional[str] = None
    llm_response: Optional[str] = None
    llm_parsed: Optional[Dict[str, Any]] = None
    
    # Decisions
    decision: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    
    # Errors
    error: Optional[str] = None
    error_details: Optional[str] = None


@dataclass  
class WorkflowLog:
    """Complete log of a workflow execution"""
    ticket_id: str
    started_at: str
    ended_at: Optional[str] = None
    total_duration_seconds: float = 0.0
    
    # Ticket info
    ticket_subject: str = ""
    ticket_text: str = ""
    ticket_images: List[str] = field(default_factory=list)
    attachment_count: int = 0
    
    # Node executions
    nodes: List[NodeExecution] = field(default_factory=list)
    
    # Final results
    resolution_status: Optional[str] = None
    final_response: Optional[str] = None
    overall_confidence: float = 0.0
    
    # Aggregated metrics
    metrics: Dict[str, Any] = field(default_factory=dict)


def start_workflow_log(ticket_id: str) -> WorkflowLog:
    """Initialize a new workflow log for a ticket (thread-safe)"""
    thread_id = threading.get_ident()
    
    with _logs_lock:
        log = WorkflowLog(
            ticket_id=ticket_id,
            started_at=datetime.now().isoformat()
        )
        _workflow_logs[thread_id] = log
        logger.info(f"📝 Started detailed logging for ticket #{ticket_id} (thread: {thread_id})")
        return log


def get_current_log() -> Optional[WorkflowLog]:
    """Get the current workflow log for this thread"""
    thread_id = threading.get_ident()
    with _logs_lock:
        return _workflow_logs.get(thread_id)


def log_node_start(node_name: str, input_summary: Dict[str, Any] = None) -> NodeExecution:
    """Log the start of a node execution (thread-safe)"""
    thread_id = threading.get_ident()
    
    node = NodeExecution(
        node_name=node_name,
        start_time=time.time(),
        input_summary=input_summary or {}
    )
    
    with _logs_lock:
        current_log = _workflow_logs.get(thread_id)
        if current_log:
            current_log.nodes.append(node)
    
    return node


def log_node_complete(
    node: NodeExecution,
    output_summary: Dict[str, Any] = None,
    retrieval_results: List[Dict[str, Any]] = None,
    llm_prompt: str = None,
    llm_response: str = None,
    llm_parsed: Dict[str, Any] = None,
    decision: Dict[str, Any] = None,
    reasoning: str = None,
    error: str = None
):
    """Log the completion of a node execution"""
    node.end_time = time.time()
    node.duration_seconds = node.end_time - node.start_time
    node.status = "error" if error else "completed"
    
    if output_summary:
        node.output_summary = output_summary
    if retrieval_results:
        node.retrieval_results = retrieval_results
    if llm_prompt:
        node.llm_prompt = llm_prompt
    if llm_response:
        node.llm_response = llm_response
    if llm_parsed:
        node.llm_parsed = llm_parsed
    if decision:
        node.decision = decision
    if reasoning:
        node.reasoning = reasoning
    if error:
        node.error = error


def log_vision_results(node: NodeExecution, results: List[Dict[str, Any]]):
    """Log vision pipeline results with full metadata"""
    formatted_results = []
    for i, match in enumerate(results):
        formatted_results.append({
            "rank": i + 1,
            "score": match.get("score", 0),
            "score_percent": f"{match.get('score', 0) * 100:.1f}%",
            "product_id": match.get("metadata", {}).get("product_id", "N/A"),
            "product_name": match.get("metadata", {}).get("product_name", "N/A"),
            "image_name": match.get("metadata", {}).get("image_name", "N/A"),
            "category": match.get("metadata", {}).get("category", "N/A"),
            "full_metadata": match.get("metadata", {})
        })
    
    node.retrieval_results = formatted_results
    node.output_summary["vision_matches"] = len(results)
    node.output_summary["top_score"] = results[0].get("score", 0) if results else 0


def log_text_rag_results(node: NodeExecution, results: List[Dict[str, Any]], gemini_answer: str = None):
    """Log text RAG pipeline results"""
    formatted_results = []
    for i, doc in enumerate(results):
        formatted_results.append({
            "rank": i + 1,
            "score": doc.get("score", 0),
            "title": doc.get("title", "N/A"),
            "source": doc.get("source", "N/A"),
            "content_preview": doc.get("content", "")[:500] + "..." if doc.get("content") else "",
            "full_content": doc.get("content", "")
        })
    
    node.retrieval_results = formatted_results
    node.output_summary["document_count"] = len(results)
    if gemini_answer:
        node.llm_response = gemini_answer


def log_past_tickets_results(node: NodeExecution, results: List[Dict[str, Any]]):
    """Log past tickets search results"""
    formatted_results = []
    for i, ticket in enumerate(results):
        metadata = ticket.get("metadata", {})
        formatted_results.append({
            "rank": i + 1,
            "score": ticket.get("score", 0),
            "ticket_id": metadata.get("ticket_id", "N/A"),
            "subject": metadata.get("subject", "N/A"),
            "resolution": metadata.get("resolution", "N/A"),
            "category": metadata.get("category", "N/A"),
            "content_preview": metadata.get("content", "")[:300] + "..." if metadata.get("content") else ""
        })
    
    node.retrieval_results = formatted_results
    node.output_summary["past_tickets_found"] = len(results)


def log_llm_interaction(
    node: NodeExecution,
    system_prompt: str,
    user_prompt: str,
    response: str,
    parsed_response: Dict[str, Any] = None
):
    """Log an LLM interaction with prompts and responses"""
    node.llm_prompt = f"=== SYSTEM ===\n{system_prompt}\n\n=== USER ===\n{user_prompt}"
    node.llm_response = response
    if parsed_response:
        node.llm_parsed = parsed_response


def complete_workflow_log(
    resolution_status: str,
    final_response: str = None,
    overall_confidence: float = 0.0,
    metrics: Dict[str, Any] = None
):
    """Complete the workflow log and save to file (thread-safe)"""
    thread_id = threading.get_ident()
    
    with _logs_lock:
        if thread_id not in _workflow_logs:
            logger.warning("No active workflow log to complete for this thread")
            return None
        
        current_log = _workflow_logs[thread_id]
        current_log.ended_at = datetime.now().isoformat()
        current_log.resolution_status = resolution_status
        current_log.final_response = final_response
        current_log.overall_confidence = overall_confidence
        current_log.metrics = metrics or {}
        
        # Calculate total duration
        start = datetime.fromisoformat(current_log.started_at)
        end = datetime.fromisoformat(current_log.ended_at)
        current_log.total_duration_seconds = (end - start).total_seconds()
        
        # Log summary to console instead of saving to file
        logger.info(f"📊 Workflow completed for ticket #{current_log.ticket_id}")
        logger.info(f"   Duration: {current_log.total_duration_seconds:.2f}s")
        logger.info(f"   Resolution: {resolution_status}")
        logger.info(f"   Nodes executed: {len(current_log.nodes)}")
        
        # Remove from thread-local storage
        del _workflow_logs[thread_id]
        
        return None  # No file path returned


def save_workflow_log(log: WorkflowLog) -> Optional[Path]:
    """
    Save workflow log to JSON file.
    Currently disabled for cloud deployment - logs go to console only.
    """
    # FILE LOGGING DISABLED - using console logging for cloud deployment
    # To re-enable, uncomment the code below:
    
    # LOG_DIR.mkdir(exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"ticket_{log.ticket_id}_{timestamp}.json"
    # filepath = LOG_DIR / filename
    # ... (write to file)
    
    logger.debug(f"File logging disabled. Ticket #{log.ticket_id} log available in console.")
    return None


def _save_workflow_log_to_file(log: WorkflowLog) -> Path:
    """
    [DISABLED] Save workflow log to JSON file.
    Keep this for future use when cloud storage is configured.
    """
    LOG_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ticket_{log.ticket_id}_{timestamp}.json"
    filepath = LOG_DIR / filename
    
    # Convert to dict for JSON serialization
    log_dict = {
        "ticket_id": log.ticket_id,
        "started_at": log.started_at,
        "ended_at": log.ended_at,
        "total_duration_seconds": log.total_duration_seconds,
        "ticket_info": {
            "subject": log.ticket_subject,
            "text_length": len(log.ticket_text),
            "text_preview": log.ticket_text[:500] + "..." if len(log.ticket_text) > 500 else log.ticket_text,
            "images": log.ticket_images,
            "attachment_count": log.attachment_count
        },
        "resolution": {
            "status": log.resolution_status,
            "overall_confidence": log.overall_confidence,
            "response_length": len(log.final_response) if log.final_response else 0,
            "response_preview": log.final_response[:1000] + "..." if log.final_response and len(log.final_response) > 1000 else log.final_response
        },
        "metrics": log.metrics,
        "nodes": []
    }
    
    # Add node executions
    for node in log.nodes:
        node_dict = {
            "node_name": node.node_name,
            "duration_seconds": node.duration_seconds,
            "status": node.status,
            "input_summary": node.input_summary,
            "output_summary": node.output_summary,
        }
        
        if node.retrieval_results:
            node_dict["retrieval_results"] = node.retrieval_results
        if node.llm_prompt:
            node_dict["llm_prompt"] = node.llm_prompt
        if node.llm_response:
            node_dict["llm_response"] = node.llm_response
        if node.llm_parsed:
            node_dict["llm_parsed"] = node.llm_parsed
        if node.decision:
            node_dict["decision"] = node.decision
        if node.reasoning:
            node_dict["reasoning"] = node.reasoning
        if node.error:
            node_dict["error"] = node.error
        
        log_dict["nodes"].append(node_dict)
    
    # Write to file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📁 Detailed workflow log saved: {filepath}")
    
    return filepath


def get_node_summary() -> str:
    """Get a summary of all nodes executed in current workflow (thread-safe)"""
    thread_id = threading.get_ident()
    current_log = _workflow_logs.get(thread_id)
    
    if not current_log:
        return "No active workflow"
    
    lines = [f"Workflow for Ticket #{current_log.ticket_id}:"]
    for node in current_log.nodes:
        status_icon = "✅" if node.status == "completed" else "❌" if node.status == "error" else "🔄"
        lines.append(f"  {status_icon} {node.node_name}: {node.duration_seconds:.2f}s")
    
    return "\n".join(lines)
