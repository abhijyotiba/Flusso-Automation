"""
Past Tickets Search Tool - FIXED VERSION
Handles both query and product_model parameters
"""

import logging
from typing import Dict, Any, Optional, List
from langchain.tools import tool

from app.clients.embeddings import embed_text
from app.clients.pinecone_client import get_pinecone_client

logger = logging.getLogger(__name__)


@tool
def past_tickets_search_tool(
    query: Optional[str] = None,
    product_model: Optional[str] = None,
    product_model_number: Optional[str] = None,  # ADDED: Alternative parameter name
    issue_type: Optional[str] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Search for similar resolved tickets from ticket history.
    
    Use this tool when:
    - You've identified the product and want to see common issues
    - Looking for resolution patterns
    - Customer reports a specific problem (leak, broken part, etc.)
    - Want to learn from past successful resolutions
    
    Args:
        query: Description of the customer's issue (e.g., "shower head leaking", 
               "missing installation parts")
        product_model: Specific product model if known (e.g., "HS6270MB")
        product_model_number: Alternative parameter name for product model
        issue_type: Type of issue if identifiable (e.g., "leak", "installation", 
                    "missing_parts", "warranty")
        top_k: Number of similar tickets to return (default: 5)
    
    Returns:
        {
            "success": bool,
            "tickets": [
                {
                    "ticket_id": int,
                    "subject": str,
                    "issue_summary": str,
                    "resolution_summary": str,
                    "resolution_type": str,
                    "product_model": str,
                    "similarity_score": float,
                    "outcome": "resolved" | "replaced" | "refunded"
                }
            ],
            "common_patterns": [str],
            "count": int,
            "message": str
        }
    """
    # CRITICAL FIX: Handle multiple parameter names for product model
    product_model = product_model or product_model_number
    
    # CRITICAL FIX: Require at least one parameter
    if not query and not product_model:
        return {
            "success": False,
            "tickets": [],
            "common_patterns": [],
            "count": 0,
            "message": "Either 'query' or 'product_model' parameter is required"
        }
    
    # If only product_model provided, build a generic query
    if not query and product_model:
        query = f"Issues with product {product_model}"
    
    logger.info(f"[PAST_TICKETS] Query: '{query}', Product: {product_model}, Issue: {issue_type}")
    
    try:
        client = get_pinecone_client()
        
        # Build search query
        search_text = query
        if product_model:
            search_text = f"Product: {product_model}. Issue: {query}"
        if issue_type:
            search_text += f" (Type: {issue_type})"
        
        # Generate embedding
        vector = embed_text(search_text)
        
        # Build metadata filter if product specified
        filter_dict = None
        if product_model:
            filter_dict = {"product_model": {"$eq": product_model.upper()}}
        
        # Query Pinecone tickets index
        results = client.query_past_tickets(vector=vector, top_k=top_k, filter_dict=filter_dict)
        
        if not results:
            return {
                "success": False,
                "tickets": [],
                "common_patterns": [],
                "count": 0,
                "message": "No similar past tickets found"
            }
        
        # Format results
        tickets = []
        resolution_types = []
        
        for result in results:
            metadata = result.get("metadata", {})
            
            ticket = {
                "ticket_id": metadata.get("ticket_id", "Unknown"),
                "subject": metadata.get("subject", "No subject"),
                "issue_summary": metadata.get("issue_summary", metadata.get("content", ""))[:300],
                "resolution_summary": metadata.get("resolution", metadata.get("resolution_summary", ""))[:300],
                "resolution_type": metadata.get("resolution_type", "resolved"),
                "product_model": metadata.get("product_model", "Unknown"),
                "category": metadata.get("category", "general"),
                "similarity_score": round(result.get("score", 0) * 100),
                "outcome": metadata.get("outcome", "resolved")
            }
            
            tickets.append(ticket)
            
            # Track resolution types
            if ticket["resolution_type"]:
                resolution_types.append(ticket["resolution_type"])
        
        # Identify common patterns
        common_patterns = _identify_patterns(tickets, resolution_types)
        
        logger.info(f"[PAST_TICKETS] Found {len(tickets)} similar ticket(s)")
        
        return {
            "success": True,
            "tickets": tickets,
            "common_patterns": common_patterns,
            "count": len(tickets),
            "message": f"Found {len(tickets)} similar resolved ticket(s)"
        }
        
    except Exception as e:
        logger.error(f"[PAST_TICKETS] Error: {e}", exc_info=True)
        return {
            "success": False,
            "tickets": [],
            "common_patterns": [],
            "count": 0,
            "message": f"Past tickets search failed: {str(e)}"
        }


def _identify_patterns(tickets: List[Dict], resolution_types: List[str]) -> List[str]:
    """Identify common patterns from similar tickets"""
    patterns = []
    
    # Most common resolution type
    if resolution_types:
        from collections import Counter
        most_common = Counter(resolution_types).most_common(1)
        if most_common:
            res_type, count = most_common[0]
            if count >= 2:
                patterns.append(f"Most common resolution: {res_type} ({count}/{len(tickets)} tickets)")
    
    # Check for recurring issues
    issue_keywords = {}
    for ticket in tickets:
        text = (ticket.get("issue_summary", "") + " " + ticket.get("subject", "")).lower()
        
        for keyword in ["leak", "broken", "missing", "install", "warranty", "replace"]:
            if keyword in text:
                issue_keywords[keyword] = issue_keywords.get(keyword, 0) + 1
    
    # Add recurring issue patterns
    for keyword, count in issue_keywords.items():
        if count >= 2:
            patterns.append(f"Recurring issue: {keyword} ({count}/{len(tickets)} tickets)")
    
    return patterns[:3]  # Return top 3 patterns