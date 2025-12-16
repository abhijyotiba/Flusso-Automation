"""
FastAPI Main Application with ReACT Agent
Webhook endpoint for Freshdesk ticket automation using intelligent ReACT loop
"""

import logging
import hashlib
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from diskcache import Cache

from app.graph.graph_builder_react import build_react_graph
from app.graph.state import TicketState
from app.utils.pii_masker import mask_email, mask_name

# ---------------------------------------------------
# LOGGING CONFIG
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

graph = None  # Global graph instance
webhook_cache = None  # Deduplication cache

# ReACT agent has more iterations, so longer timeout
WORKFLOW_TIMEOUT = 600  # 10 minutes


# ---------------------------------------------------
# LIFESPAN HOOKS
# ---------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph, webhook_cache
    logger.info("🚀 Starting Flusso Workflow Automation (ReACT Agent Mode)...")

    # Initialize deduplication cache
    webhook_cache = Cache(".cache/webhook_dedup")
    logger.info("✅ Webhook deduplication cache initialized")

    graph = build_react_graph()
    logger.info("✅ LangGraph ReACT workflow initialized")

    yield

    # Cleanup
    if webhook_cache:
        webhook_cache.close()
    logger.info("🛑 Shutting down Flusso Workflow Automation...")


# ---------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------
app = FastAPI(
    title="Flusso Workflow Automation (ReACT)",
    description="Intelligent ReACT Agent for Freshdesk ticket automation",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for Freshdesk webhook calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------
# HEALTH ENDPOINTS
# ---------------------------------------------------
@app.get("/")
@app.head("/")
async def root():
    return {
        "service": "Flusso Workflow Automation (ReACT Agent)",
        "status": "running",
        "mode": "react_agent",
        "graph_initialized": graph is not None,
    }


@app.get("/favicon.ico")
async def favicon():
    """Return empty response for favicon requests to avoid 404 spam in logs."""
    return Response(status_code=204)


@app.get("/health")
@app.head("/health")
async def health_check():
    """Basic health check - returns quickly."""
    return {
        "status": "healthy",
        "mode": "react_agent",
        "graph_ready": graph is not None,
    }


@app.get("/health/deep")
async def deep_health_check():
    """Deep health check - validates all components."""
    status = {
        "status": "healthy",
        "mode": "react_agent",
        "components": {
            "graph": graph is not None,
            "cache": webhook_cache is not None,
        }
    }
    
    # Check if graph has expected nodes
    if graph:
        try:
            # Verify key nodes exist
            status["components"]["graph_nodes"] = True
        except Exception as e:
            status["components"]["graph_nodes"] = False
            status["components"]["graph_error"] = str(e)
    
    return status


# ---------------------------------------------------
# WEBHOOK DEDUPLICATION HELPERS
# ---------------------------------------------------
def _create_webhook_key(ticket_id: str, updated_at: str = None) -> str:
    """Create unique key for webhook deduplication."""
    data = f"{ticket_id}:{updated_at or 'unknown'}"
    return hashlib.md5(data.encode()).hexdigest()


def _is_duplicate_webhook(key: str, ttl_seconds: int = 30) -> bool:
    """Check if this webhook was recently processed."""
    if key in webhook_cache:
        return True
    webhook_cache.set(key, True, expire=ttl_seconds)
    return False


# ---------------------------------------------------
# MAIN WEBHOOK ENDPOINT
# ---------------------------------------------------
@app.post("/webhook")
async def freshdesk_webhook(request: Request):
    """
    Main webhook endpoint for Freshdesk ticket events.
    Uses ReACT agent for intelligent information gathering.
    """
    global graph

    if not graph:
        logger.error("Graph not initialized!")
        raise HTTPException(status_code=503, detail="Workflow graph not ready")

    try:
        # Parse request body
        body = await request.json()
        logger.info(f"📬 Webhook received: {body}")

        # Extract ticket_id from various formats
        ticket_id = None
        updated_at = None

        # Format 1: Direct ticket_id
        if "ticket_id" in body:
            ticket_id = str(body["ticket_id"])
        # Format 2: freshdesk_webhook.ticket_id
        elif "freshdesk_webhook" in body:
            fd_data = body["freshdesk_webhook"]
            ticket_id = str(fd_data.get("ticket_id"))
            updated_at = fd_data.get("ticket_updated_at")
        # Format 3: Nested ticket object
        elif "ticket" in body:
            ticket_id = str(body["ticket"].get("id"))
            updated_at = body["ticket"].get("updated_at")

        if not ticket_id:
            logger.warning("No ticket_id found in webhook payload")
            return JSONResponse(
                status_code=400,
                content={"error": "Missing ticket_id in payload"}
            )

        # Deduplication check
        webhook_key = _create_webhook_key(ticket_id, updated_at)
        if _is_duplicate_webhook(webhook_key, ttl_seconds=30):
            logger.info(f"🔄 Duplicate webhook for ticket {ticket_id}, skipping")
            return JSONResponse(
                status_code=200,
                content={"status": "skipped", "reason": "duplicate_webhook", "ticket_id": ticket_id}
            )

        logger.info(f"🎫 Processing ticket #{ticket_id} with ReACT agent...")

        # Initialize state
        initial_state: TicketState = {
            "ticket_id": ticket_id,
            "audit_events": [],
            # ReACT-specific initialization
            "react_iterations": [],
            "react_total_iterations": 0,
            "react_status": "pending",
            "gathered_documents": [],
            "gathered_images": [],
            "gathered_past_tickets": [],
        }

        # Run the ReACT workflow
        try:
            import asyncio
            final_state = await asyncio.wait_for(
                asyncio.to_thread(graph.invoke, initial_state),
                timeout=WORKFLOW_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"⏰ Workflow timeout after {WORKFLOW_TIMEOUT}s for ticket #{ticket_id}")
            return JSONResponse(
                status_code=504,
                content={
                    "status": "timeout",
                    "ticket_id": ticket_id,
                    "timeout_seconds": WORKFLOW_TIMEOUT
                }
            )

        # Extract key results
        resolution = final_state.get("resolution_decision", "unknown")
        react_iterations = final_state.get("react_total_iterations", 0)
        react_status = final_state.get("react_status", "unknown")
        product_identified = final_state.get("identified_product") is not None
        
        response_data = {
            "status": "success",
            "ticket_id": ticket_id,
            "resolution": resolution,
            "mode": "react_agent",
            "react_iterations": react_iterations,
            "react_status": react_status,
            "product_identified": product_identified,
            "documents_found": len(final_state.get("gathered_documents", [])),
            "images_found": len(final_state.get("gathered_images", [])),
            "past_tickets_found": len(final_state.get("gathered_past_tickets", [])),
        }

        # Log PII-masked summary
        requester = final_state.get("requester_email", "")
        masked_email = mask_email(requester) if requester else "N/A"
        logger.info(f"✅ Ticket #{ticket_id} processed: {resolution} | ReACT: {react_iterations} iterations | Requester: {masked_email}")

        return JSONResponse(status_code=200, content=response_data)

    except Exception as e:
        logger.error(f"❌ Webhook processing error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal processing error", "detail": str(e)}
        )


# ---------------------------------------------------
# DEBUG ENDPOINTS
# ---------------------------------------------------
@app.post("/debug/process/{ticket_id}")
async def debug_process_ticket(ticket_id: str, dry_run: bool = False):
    """
    Debug endpoint to manually process a ticket.
    Set dry_run=True to test without updating Freshdesk.
    """
    global graph

    if not graph:
        raise HTTPException(status_code=503, detail="Workflow graph not ready")

    logger.info(f"🔧 Debug processing ticket #{ticket_id} (dry_run={dry_run})")

    initial_state: TicketState = {
        "ticket_id": ticket_id,
        "audit_events": [],
        "react_iterations": [],
        "react_total_iterations": 0,
        "react_status": "pending",
        "gathered_documents": [],
        "gathered_images": [],
        "gathered_past_tickets": [],
    }

    if dry_run:
        # Add flag to skip Freshdesk update
        initial_state["skip_freshdesk_update"] = True

    try:
        import asyncio
        final_state = await asyncio.wait_for(
            asyncio.to_thread(graph.invoke, initial_state),
            timeout=WORKFLOW_TIMEOUT
        )

        # Extract ReACT reasoning chain for debugging
        react_chain = []
        for iteration in final_state.get("react_iterations", []):
            react_chain.append({
                "iteration": iteration.get("iteration"),
                "thought": iteration.get("thought", "")[:200],
                "action": iteration.get("action"),
                "observation": iteration.get("observation", "")[:200],
                "duration": iteration.get("duration", 0)
            })

        return {
            "status": "success",
            "ticket_id": ticket_id,
            "dry_run": dry_run,
            "react_status": final_state.get("react_status"),
            "react_iterations": final_state.get("react_total_iterations"),
            "react_chain": react_chain,
            "identified_product": final_state.get("identified_product"),
            "gathered_documents": len(final_state.get("gathered_documents", [])),
            "gathered_images": len(final_state.get("gathered_images", [])),
            "gathered_past_tickets": len(final_state.get("gathered_past_tickets", [])),
            "resolution_decision": final_state.get("resolution_decision"),
            "generated_reply": final_state.get("generated_reply", "")[:500] if final_state.get("generated_reply") else None,
            "audit_events_count": len(final_state.get("audit_events", [])),
        }

    except asyncio.TimeoutError:
        return {
            "status": "timeout",
            "ticket_id": ticket_id,
            "timeout_seconds": WORKFLOW_TIMEOUT
        }
    except Exception as e:
        logger.error(f"Debug processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/react-iterations/{ticket_id}")
async def get_react_iterations(ticket_id: str):
    """
    Get detailed ReACT iterations for a specific ticket.
    Useful for debugging agent reasoning.
    """
    # This would need access to stored audit logs
    # For now, return a placeholder
    return {
        "message": "ReACT iteration history not persisted",
        "suggestion": "Check workflow_logs/ for detailed audit trails"
    }


# ---------------------------------------------------
# COMPARISON ENDPOINT (Sequential vs ReACT)
# ---------------------------------------------------
@app.get("/info")
async def get_workflow_info():
    """Return information about the ReACT workflow configuration."""
    return {
        "mode": "react_agent",
        "version": "2.0.0",
        "max_iterations": 15,
        "timeout_seconds": WORKFLOW_TIMEOUT,
        "available_tools": [
            "product_search_tool",
            "document_search_tool",
            "vision_search_tool",
            "past_tickets_search_tool",
            "attachment_analyzer_tool",
            "finish_tool"
        ],
        "workflow_flow": [
            "fetch_ticket",
            "routing",
            "react_agent (loops with evidence_resolver)",
            "customer_lookup",
            "vip_rules",
            "draft_response",
            "vip_compliance",
            "resolution_logic",
            "freshdesk_update",
            "audit_log"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main_react:app", host="0.0.0.0", port=8000, reload=True)
