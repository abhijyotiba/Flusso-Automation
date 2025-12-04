"""
FastAPI Main Application
Webhook endpoint for Freshdesk ticket automation
"""

import logging
import hashlib
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from diskcache import Cache

from app.graph.graph_builder import build_graph
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


# ---------------------------------------------------
# LIFESPAN HOOKS
# ---------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph, webhook_cache
    logger.info("🚀 Starting Flusso Workflow Automation...")

    # Initialize deduplication cache
    webhook_cache = Cache(".cache/webhook_dedup")
    logger.info("✅ Webhook deduplication cache initialized")

    graph = build_graph()
    logger.info("✅ LangGraph workflow initialized")

    yield

    # Cleanup
    if webhook_cache:
        webhook_cache.close()
    logger.info("🛑 Shutting down Flusso Workflow Automation...")


# ---------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------
app = FastAPI(
    title="Flusso Workflow Automation",
    description="Multimodal RAG for Freshdesk ticket automation",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Freshdesk webhook calls
# Note: When using allow_origins=["*"], credentials should be False for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Freshdesk uses public IPs
    allow_credentials=False,  # Must be False when allow_origins is wildcard
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------
# HEALTH ENDPOINTS
# ---------------------------------------------------
@app.get("/")
async def root():
    return {
        "service": "Flusso Workflow Automation",
        "status": "running",
        "graph_initialized": graph is not None,
    }


@app.get("/health")
async def health_check():
    """Basic health check - returns quickly."""
    return {
        "status": "healthy",
        "graph_ready": graph is not None,
        "service": "Flusso Workflow Automation",
    }


@app.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check that verifies all external dependencies.
    Use for monitoring/alerting systems.
    """
    from app.config.settings import settings
    import httpx
    
    health_status = {
        "status": "healthy",
        "service": "Flusso Workflow Automation",
        "graph_ready": graph is not None,
        "dependencies": {}
    }
    
    # Check Freshdesk API
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{settings.FRESHDESK_DOMAIN}/api/v2/tickets",
                auth=(settings.FRESHDESK_API_KEY, "X"),
                params={"per_page": 1}
            )
            health_status["dependencies"]["freshdesk"] = {
                "status": "healthy" if resp.status_code == 200 else "degraded",
                "response_code": resp.status_code
            }
    except Exception as e:
        health_status["dependencies"]["freshdesk"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Pinecone
    try:
        from app.clients.pinecone_client import get_pinecone_client
        pc_client = get_pinecone_client()
        # Just verify client exists (actual index check would be too slow)
        health_status["dependencies"]["pinecone"] = {
            "status": "healthy" if pc_client else "unhealthy"
        }
    except Exception as e:
        health_status["dependencies"]["pinecone"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Gemini
    try:
        from google import genai
        from app.config.settings import settings
        # Verify API key exists and client can be created
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        health_status["dependencies"]["gemini"] = {
            "status": "healthy" if settings.GEMINI_API_KEY and client else "unhealthy"
        }
    except Exception as e:
        health_status["dependencies"]["gemini"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check webhook deduplication cache
    health_status["dependencies"]["webhook_cache"] = {
        "status": "healthy" if webhook_cache else "unhealthy"
    }
    
    return health_status


# ---------------------------------------------------
# FRESHDESK WEBHOOK (with deduplication)
# ---------------------------------------------------
@app.post("/freshdesk/webhook")
async def freshdesk_webhook(request: Request):
    """
    Freshdesk webhook endpoint
    Receives ticket creation/update events.

    Expected payload:
    {
        "ticket_id": 12345,
        "freshdesk_webhook": {...}
    }
    """

    global graph, webhook_cache
    if graph is None:
        logger.warning("⚠ Graph was not initialized — rebuilding...")
        graph = build_graph()

    # Initialize dedup_key to None to avoid NameError in exception handler
    dedup_key = None
    
    try:
        payload = await request.json()
        ticket_id = payload.get("ticket_id")
        
        if not ticket_id:
            raise HTTPException(status_code=400, detail="Missing ticket_id in payload")

        # ---------------------------------------------------
        # DEDUPLICATION CHECK
        # Prevent processing same ticket update multiple times
        # ---------------------------------------------------
        updated_at = payload.get("updated_at") or payload.get("freshdesk_webhook", {}).get("ticket_updated_at", "")
        dedup_key = hashlib.sha256(f"{ticket_id}:{updated_at}".encode()).hexdigest()
        
        if webhook_cache and webhook_cache.get(dedup_key):
            logger.warning(f"⚠️ Duplicate webhook detected for ticket #{ticket_id}, skipping")
            return JSONResponse(
                content={"status": "duplicate", "ticket_id": ticket_id, "message": "Already processed"},
                status_code=200
            )
        
        # Mark as processing (expires in 1 hour)
        if webhook_cache:
            webhook_cache.set(dedup_key, True, expire=3600)
        
        logger.info(f"📨 Received webhook for ticket #{ticket_id}")

        # ---------------------------------------------------
        # Initialize minimal safe TicketState
        # ALL REQUIRED FIELDS MUST EXIST
        # ---------------------------------------------------

        initial_state: TicketState = {
            "ticket_id": str(ticket_id),

            # Raw incoming data (optional field)
            "freshdesk_webhook_payload": payload,

            # Required defaults for workflow stability
            "ticket_subject": "",
            "ticket_text": "",
            "ticket_images": [],
            "requester_email": "",
            "requester_name": "",
            "ticket_type": None,
            "priority": None,
            "tags": [],
            "created_at": None,
            "updated_at": None,

            # Routing flags
            "has_text": False,
            "has_image": False,

            # Customer info
            "customer_type": None,
            "customer_metadata": {},
            "vip_rules": {},

            # RAG results
            "text_retrieval_results": [],
            "image_retrieval_results": [],
            "past_ticket_results": [],
            "multimodal_context": "",

            # Decision values
            "product_match_confidence": 0.0,
            "hallucination_risk": 0.0,
            "enough_information": False,
            "vip_compliant": True,

            # Response
            "clarification_message": None,
            "draft_response": None,
            "final_response_public": None,
            "final_private_note": None,
            "resolution_status": None,
            "extra_tags": [],

            # Audit trail
            "audit_events": [
                {"event": "webhook_received", "ticket_id": ticket_id}
            ],
        }

        logger.info(f"🔄 Starting workflow for ticket #{ticket_id}")

        final_state = graph.invoke(initial_state)

        result = {
            "ticket_id": ticket_id,
            "resolution_status": final_state.get("resolution_status"),
            "category": final_state.get("ticket_category"),
            "customer_type": final_state.get("customer_type"),
            "tags": final_state.get("tags"),
            "workflow_completed": True,
        }

        logger.info(
            f"✅ Workflow completed for ticket #{ticket_id}: {result['resolution_status']}"
        )

        return JSONResponse(content=result, status_code=200)

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"❌ Workflow error: {e}", exc_info=True)
        # Remove from dedup cache on error so it can be retried
        if webhook_cache and dedup_key:
            webhook_cache.delete(dedup_key)
        return JSONResponse(
            content={"error": str(e), "workflow_completed": False},
            status_code=500,
        )


# ---------------------------------------------------
# SIMPLE WEBHOOK (alias for testing)
# ---------------------------------------------------
@app.post("/webhook")
async def simple_webhook(request: Request):
    """
    Simple webhook endpoint (alias for /freshdesk/webhook)
    """
    return await freshdesk_webhook(request)


# ---------------------------------------------------
# LOCAL DEV ENTRY POINT
# ---------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
