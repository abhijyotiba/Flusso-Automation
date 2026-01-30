"""
Freshdesk API Client
Clean, Correct & Production-Ready Version
"""

import requests
import logging
import time
from typing import List, Dict, Optional, Any
from requests.auth import HTTPBasicAuth

from app.config.settings import settings
from app.utils.retry import retry_api_call, TRANSIENT_EXCEPTIONS
from app.utils.pii_masker import mask_api_key

logger = logging.getLogger(__name__)


class FreshdeskClient:
    """
    Service for interacting with Freshdesk API.
    Implemented cleanly for ticket retrieval, updates, and notes.
    """

    def __init__(self):
        domain = settings.freshdesk_domain
        domain = domain.replace("https://", "").replace("http://", "")
        domain = domain.split("/")[0]

        api_key = settings.freshdesk_api_key.strip()

        self.domain = domain
        self.api_key = api_key
        self.base_url = f"https://{self.domain}/api/v2"
        self.auth = HTTPBasicAuth(self.api_key, "X")
        self.timeout = 30

        self.headers = {"Content-Type": "application/json"}

        # Log with masked API key for security
        logger.info(f"Freshdesk client initialized → {self.base_url} (key: {mask_api_key(api_key)})")

    # --------------------------------------------------------------------
    # Rate Limit Handler
    # --------------------------------------------------------------------
    def _handle_rate_limit(self, response: requests.Response):
        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", 60))
            logger.warning(f"[Freshdesk] Rate limited. Waiting {wait}s")
            time.sleep(wait)

    # --------------------------------------------------------------------
    # GET Ticket (with retry)
    # --------------------------------------------------------------------
    @retry_api_call
    def get_ticket(self, ticket_id: int, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Fetch a Freshdesk ticket by ID"""
        url = f"{self.base_url}/tickets/{ticket_id}"
        params = params or {}

        response = requests.get(
            url,
            auth=self.auth,
            params=params,
            headers=self.headers,
            timeout=self.timeout
        )
        self._handle_rate_limit(response)
        response.raise_for_status()
        return response.json()

    # --------------------------------------------------------------------
    # GET Conversations (with retry)
    # --------------------------------------------------------------------
    @retry_api_call
    def get_ticket_conversations(self, ticket_id: int) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/tickets/{ticket_id}/conversations"

        response = requests.get(
            url,
            auth=self.auth,
            headers=self.headers,
            timeout=self.timeout
        )
        self._handle_rate_limit(response)
        response.raise_for_status()
        return response.json()

    # --------------------------------------------------------------------
    # Add Note (with retry)
    # --------------------------------------------------------------------
    @retry_api_call
    def add_note(self, ticket_id: int, body: str, private: bool = True) -> Dict[str, Any]:
        url = f"{self.base_url}/tickets/{ticket_id}/notes"
        payload = {"body": body, "private": private}

        response = requests.post(
            url,
            json=payload,
            auth=self.auth,
            headers=self.headers,
            timeout=self.timeout,
        )
        self._handle_rate_limit(response)
        response.raise_for_status()
        time.sleep(0.4)
        return response.json()

    # --------------------------------------------------------------------
    # Update Ticket (with retry)
    # --------------------------------------------------------------------
    @retry_api_call
    def update_ticket(self, ticket_id: int, **fields) -> Dict[str, Any]:
        url = f"{self.base_url}/tickets/{ticket_id}"

        response = requests.put(
            url,
            json=fields,
            auth=self.auth,
            headers=self.headers,
            timeout=self.timeout,
        )
        self._handle_rate_limit(response)
        response.raise_for_status()
        time.sleep(0.4)
        return response.json()

    # --------------------------------------------------------------------
    # Extract Ticket Fields (Normalized)
    # --------------------------------------------------------------------
    def extract_ticket_data(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        requester = ticket.get("requester") or {}
        description = ticket.get("description_text") or ticket.get("description") or ""

        return {
            "ticket_id": ticket.get("id"),
            "subject": ticket.get("subject", ""),
            "description": description,
            "status": ticket.get("status"),
            "priority": ticket.get("priority"),
            "requester_name": requester.get("name") or "Unknown",
            "requester_email": requester.get("email") or "",
            "created_at": ticket.get("created_at"),
            "updated_at": ticket.get("updated_at"),
            "tags": ticket.get("tags", []),
            "type": ticket.get("type"),
            "attachments": ticket.get("attachments", [])
        }


# Singleton client
_client = None


def get_freshdesk_client() -> FreshdeskClient:
    global _client
    if _client is None:
        _client = FreshdeskClient()
    return _client
