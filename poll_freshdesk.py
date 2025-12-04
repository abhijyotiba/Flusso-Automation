"""
Freshdesk Ticket Poller - Local Development
Continuously polls Freshdesk for new tickets and processes them through the workflow.

Usage:
    python poll_freshdesk.py

This script:
1. Checks for new/updated tickets every 30 seconds
2. Automatically runs the workflow for each new ticket
3. Tracks processed tickets to avoid duplicates
"""

import sys
import os
import time
import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import requests
from app.config.settings import settings
from app.graph.graph_builder import build_graph
from app.graph.state import TicketState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Polling configuration
POLL_INTERVAL = 30  # seconds
LOOKBACK_MINUTES = 60  # Check tickets from last hour
PROCESSED_FILE = ".cache/processed_tickets.json"


class FreshdeskPoller:
    """Polls Freshdesk for new tickets and processes them"""
    
    def __init__(self):
        self.graph = None
        self.processed_tickets = self._load_processed()
        self.freshdesk_url = f"https://{settings.freshdesk_domain}/api/v2"
        self.auth = (settings.freshdesk_api_key, "X")
        
    def _load_processed(self) -> dict:
        """Load previously processed ticket IDs"""
        Path(".cache").mkdir(exist_ok=True)
        try:
            if os.path.exists(PROCESSED_FILE):
                with open(PROCESSED_FILE, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {}
    
    def _save_processed(self):
        """Save processed ticket IDs"""
        with open(PROCESSED_FILE, 'w') as f:
            json.dump(self.processed_tickets, f)
    
    def _get_ticket_hash(self, ticket: dict) -> str:
        """Generate unique hash for ticket state"""
        key = f"{ticket['id']}:{ticket.get('updated_at', '')}:{ticket.get('status', '')}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def initialize(self):
        """Initialize the workflow graph"""
        logger.info("🚀 Initializing Flusso Workflow...")
        self.graph = build_graph()
        logger.info("✅ Workflow graph ready")
    
    def fetch_recent_tickets(self) -> list:
        """Fetch tickets created/updated in the last hour"""
        try:
            # Get tickets updated recently
            params = {
                "order_by": "updated_at",
                "order_type": "desc",
                "per_page": 30
            }
            
            response = requests.get(
                f"{self.freshdesk_url}/tickets",
                auth=self.auth,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            tickets = response.json()
            
            # Filter to recent tickets only
            cutoff = datetime.utcnow() - timedelta(minutes=LOOKBACK_MINUTES)
            recent = []
            
            for ticket in tickets:
                updated_at = ticket.get('updated_at', '')
                if updated_at:
                    # Parse ISO format
                    try:
                        ticket_time = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                        if ticket_time.replace(tzinfo=None) > cutoff:
                            recent.append(ticket)
                    except:
                        recent.append(ticket)  # Include if can't parse
            
            return recent
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch tickets: {e}")
            return []
    
    def process_ticket(self, ticket: dict) -> dict:
        """Process a single ticket through the workflow"""
        ticket_id = ticket['id']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎫 Processing Ticket #{ticket_id}")
        logger.info(f"   Subject: {ticket.get('subject', 'N/A')[:50]}")
        logger.info(f"   Status: {ticket.get('status', 'N/A')}")
        logger.info(f"   Priority: {ticket.get('priority', 'N/A')}")
        logger.info(f"{'='*60}")
        
        # Build initial state
        initial_state: TicketState = {
            "ticket_id": str(ticket_id),
            "freshdesk_webhook_payload": {"ticket_id": ticket_id},
            
            # Required defaults
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
            "audit_events": [{"event": "poller_triggered", "ticket_id": ticket_id}],
        }
        
        try:
            start_time = time.time()
            final_state = self.graph.invoke(initial_state)
            duration = time.time() - start_time
            
            logger.info(f"\n✅ Ticket #{ticket_id} completed in {duration:.1f}s")
            logger.info(f"   Resolution: {final_state.get('resolution_status', 'N/A')}")
            logger.info(f"   Category: {final_state.get('ticket_category', 'N/A')}")
            
            return {
                "success": True,
                "ticket_id": ticket_id,
                "resolution": final_state.get('resolution_status'),
                "duration": duration
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing ticket #{ticket_id}: {e}")
            return {
                "success": False,
                "ticket_id": ticket_id,
                "error": str(e)
            }
    
    def poll_once(self) -> int:
        """Single poll iteration - returns count of new tickets processed"""
        tickets = self.fetch_recent_tickets()
        
        if not tickets:
            return 0
        
        new_count = 0
        
        for ticket in tickets:
            ticket_id = str(ticket['id'])
            ticket_hash = self._get_ticket_hash(ticket)
            
            # Check if already processed this version
            if self.processed_tickets.get(ticket_id) == ticket_hash:
                continue
            
            # New or updated ticket - process it
            new_count += 1
            result = self.process_ticket(ticket)
            
            if result.get('success'):
                # Mark as processed
                self.processed_tickets[ticket_id] = ticket_hash
                self._save_processed()
        
        return new_count
    
    def run_continuous(self):
        """Run continuous polling loop"""
        logger.info("\n" + "="*60)
        logger.info("🔄 FLUSSO FRESHDESK POLLER - CONTINUOUS MODE")
        logger.info("="*60)
        logger.info(f"📋 Polling interval: {POLL_INTERVAL} seconds")
        logger.info(f"📋 Lookback window: {LOOKBACK_MINUTES} minutes")
        logger.info(f"📋 Freshdesk domain: {settings.freshdesk_domain}")
        logger.info("="*60)
        logger.info("\n⏳ Waiting for new tickets... (Press Ctrl+C to stop)\n")
        
        self.initialize()
        
        try:
            while True:
                new_tickets = self.poll_once()
                
                if new_tickets > 0:
                    logger.info(f"\n✨ Processed {new_tickets} new ticket(s)")
                else:
                    # Show a subtle heartbeat
                    print(".", end="", flush=True)
                
                time.sleep(POLL_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("\n\n🛑 Poller stopped by user")
            logger.info(f"📊 Total tickets processed this session: {len(self.processed_tickets)}")


def main():
    """Entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║       🌊 FLUSSO FRESHDESK TICKET POLLER                  ║
    ║                                                           ║
    ║   Automatically processes new Freshdesk tickets           ║
    ║   through your AI workflow.                               ║
    ║                                                           ║
    ║   Press Ctrl+C to stop                                    ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    poller = FreshdeskPoller()
    poller.run_continuous()


if __name__ == "__main__":
    main()
