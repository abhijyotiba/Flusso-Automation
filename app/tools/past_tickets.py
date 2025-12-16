"""
Past Tickets Search Tool - IMPROVED VERSION
Focuses on issue type/scenario similarity rather than just product matching
"""

import logging
from typing import Dict, Any, Optional, List
from langchain.tools import tool

from app.clients.embeddings import embed_text
from app.clients.pinecone_client import get_pinecone_client

logger = logging.getLogger(__name__)


# Issue type keywords for better classification
ISSUE_TYPE_KEYWORDS = {
    "replacement": ["replace", "replacement", "send another", "new one", "swap", "exchange"],
    "warranty_claim": ["warranty", "guarantee", "covered", "warranty claim", "within warranty"],
    "leak": ["leak", "leaking", "drip", "dripping", "water coming out"],
    "broken": ["broken", "damaged", "cracked", "shattered", "not working"],
    "missing_parts": ["missing", "not included", "parts missing", "incomplete"],
    "installation": ["install", "setup", "assembly", "mounting", "how to"],
    "defective": ["defect", "defective", "faulty", "manufacturing issue"],
    "refund": ["refund", "money back", "return", "returns"],
}


def _extract_issue_type(text: str) -> str:
    """Extract the primary issue type from text."""
    text_lower = text.lower()
    
    for issue_type, keywords in ISSUE_TYPE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return issue_type
    
    return "general"


def _build_scenario_focused_query(query: str, product_model: Optional[str], issue_type: Optional[str]) -> str:
    """
    Build a search query that prioritizes scenario/issue type matching over product matching.
    
    Key insight: Past tickets are most useful when they show how similar ISSUES were resolved,
    not just issues with the same product. A replacement request for a faucet should find
    other replacement requests, not just other faucet issues.
    """
    # Auto-detect issue type if not provided
    detected_issue_type = issue_type or _extract_issue_type(query or "")
    
    # Build query components
    components = []
    
    # PRIORITY 1: Issue type / scenario (most important for finding relevant resolutions)
    if detected_issue_type and detected_issue_type != "general":
        issue_descriptions = {
            "replacement": "Customer requesting product replacement or exchange",
            "warranty_claim": "Warranty claim request for defective product",
            "leak": "Product leaking water or fluid",
            "broken": "Product is broken or damaged",
            "missing_parts": "Parts missing from shipment or package",
            "installation": "Help with product installation or setup",
            "defective": "Manufacturing defect or quality issue",
            "refund": "Customer requesting refund or return",
        }
        components.append(f"Issue Type: {issue_descriptions.get(detected_issue_type, detected_issue_type)}")
    
    # PRIORITY 2: Customer's actual issue description
    if query:
        components.append(f"Customer Issue: {query}")
    
    # PRIORITY 3: Product as context (but not the primary search focus)
    # Note: We don't use product_model as a filter, just as context for better semantic matching
    if product_model:
        components.append(f"(Product context: {product_model})")
    
    return " | ".join(components)


@tool
def past_tickets_search_tool(
    query: Optional[str] = None,
    product_model: Optional[str] = None,
    product_model_number: Optional[str] = None,  # Alternative parameter name
    issue_type: Optional[str] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Search for similar resolved tickets based on ISSUE TYPE/SCENARIO.
    
    IMPORTANT: This tool finds tickets with similar issues and resolutions,
    NOT just tickets for the same product. A replacement request should find
    other replacement requests, a leak complaint should find other leak resolutions.
    
    Use this tool when:
    - Looking for how similar issues were resolved in the past
    - Finding resolution patterns for specific issue types (replacement, refund, etc.)
    - Learning from past successful customer service interactions
    - Understanding common resolution workflows
    
    Args:
        query: Description of the customer's issue (e.g., "customer wants replacement 
               for broken faucet", "asking for refund due to defective product")
        product_model: Product model for context (optional, not used as primary filter)
        product_model_number: Alternative parameter name for product model
        issue_type: Type of issue (e.g., "replacement", "leak", "missing_parts", 
                    "warranty_claim", "refund", "installation", "broken", "defective")
        top_k: Number of similar tickets to return (default: 5)
    
    Returns:
        {
            "success": bool,
            "tickets": [...],
            "common_patterns": [str],
            "detected_issue_type": str,
            "count": int,
            "message": str
        }
    """
    # Handle multiple parameter names for product model
    product_model = product_model or product_model_number
    
    # Require at least query or issue_type for meaningful search
    if not query and not issue_type:
        return {
            "success": False,
            "tickets": [],
            "common_patterns": [],
            "count": 0,
            "message": "Either 'query' or 'issue_type' parameter is required. Provide a description of the issue to find similar past tickets."
        }
    
    # If only product_model provided without query, create issue-focused query
    if not query and product_model and issue_type:
        query = f"{issue_type} issue requiring resolution"
    elif not query and product_model:
        query = "Customer issue requiring resolution"
    
    # Detect issue type from query
    detected_issue_type = issue_type or _extract_issue_type(query or "")
    
    logger.info(f"[PAST_TICKETS] Query: '{query}', Product: {product_model}, "
                f"Issue Type: {issue_type}, Detected: {detected_issue_type}")
    
    try:
        client = get_pinecone_client()
        
        # IMPROVED: Build scenario-focused search query
        # This prioritizes finding similar ISSUES, not just same products
        search_text = _build_scenario_focused_query(query, product_model, detected_issue_type)
        
        logger.info(f"[PAST_TICKETS] Scenario-focused search: {search_text}")
        
        # Generate embedding
        vector = embed_text(search_text)
        
        # CHANGED: Don't filter by product_model - we want to find similar SCENARIOS
        # across all products, not just the same product. A replacement request
        # for a faucet should match other replacement requests for any product.
        # The semantic search will naturally rank product-relevant results higher
        # if the issue description mentions the product.
        filter_dict = None
        
        # Optional: Filter by issue_type if stored in metadata and explicitly requested
        # This can help narrow down to specific resolution types
        if issue_type:
            # Only apply filter if issue_type metadata exists in the index
            # filter_dict = {"issue_type": {"$eq": issue_type}}
            pass  # Disabled by default - rely on semantic search
        
        # Query Pinecone tickets index
        results = client.query_past_tickets(vector=vector, top_k=top_k, filter_dict=filter_dict)
        
        if not results:
            return {
                "success": False,
                "tickets": [],
                "common_patterns": [],
                "detected_issue_type": detected_issue_type,
                "count": 0,
                "message": f"No similar past tickets found for issue type: {detected_issue_type}"
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
                "outcome": metadata.get("outcome", "resolved"),
                "issue_type": metadata.get("issue_type", _extract_issue_type(
                    metadata.get("issue_summary", "") + " " + metadata.get("subject", "")
                ))
            }
            
            tickets.append(ticket)
            
            # Track resolution types
            if ticket["resolution_type"]:
                resolution_types.append(ticket["resolution_type"])
        
        # Identify common patterns
        common_patterns = _identify_patterns(tickets, resolution_types)
        
        logger.info(f"[PAST_TICKETS] Found {len(tickets)} similar ticket(s) for issue type: {detected_issue_type}")
        
        return {
            "success": True,
            "tickets": tickets,
            "common_patterns": common_patterns,
            "detected_issue_type": detected_issue_type,
            "count": len(tickets),
            "message": f"Found {len(tickets)} similar tickets for '{detected_issue_type}' issue type"
        }
        
    except Exception as e:
        logger.error(f"[PAST_TICKETS] Error: {e}", exc_info=True)
        return {
            "success": False,
            "tickets": [],
            "common_patterns": [],
            "detected_issue_type": detected_issue_type if 'detected_issue_type' in dir() else "unknown",
            "count": 0,
            "message": f"Past tickets search failed: {str(e)}"
        }


def _identify_patterns(tickets: List[Dict], resolution_types: List[str]) -> List[str]:
    """
    Identify common patterns from similar tickets.
    
    Focuses on:
    1. Resolution types (how issues were resolved)
    2. Common issue categories
    3. Outcome patterns (replaced, refunded, resolved)
    """
    from collections import Counter
    patterns = []
    
    # Most common resolution type
    if resolution_types:
        most_common = Counter(resolution_types).most_common(1)
        if most_common:
            res_type, count = most_common[0]
            if count >= 2:
                patterns.append(f"Most common resolution: {res_type} ({count}/{len(tickets)} tickets)")
    
    # Most common outcome
    outcomes = [t.get("outcome", "") for t in tickets if t.get("outcome")]
    if outcomes:
        most_common_outcome = Counter(outcomes).most_common(1)
        if most_common_outcome:
            outcome, count = most_common_outcome[0]
            if count >= 2:
                patterns.append(f"Common outcome: {outcome} ({count}/{len(tickets)} tickets)")
    
    # Check for recurring issue types
    issue_types = [t.get("issue_type", "") for t in tickets if t.get("issue_type")]
    if issue_types:
        most_common_issue = Counter(issue_types).most_common(1)
        if most_common_issue:
            issue, count = most_common_issue[0]
            if count >= 2:
                patterns.append(f"Similar issue type: {issue} ({count}/{len(tickets)} tickets)")
    
    # Check for recurring keywords in issues
    issue_keywords = {}
    for ticket in tickets:
        text = (ticket.get("issue_summary", "") + " " + ticket.get("subject", "")).lower()
        
        for keyword in ["replacement sent", "refund processed", "troubleshooting helped", 
                        "warranty approved", "parts shipped", "escalated"]:
            if keyword in text:
                issue_keywords[keyword] = issue_keywords.get(keyword, 0) + 1
    
    # Add recurring resolution patterns
    for keyword, count in sorted(issue_keywords.items(), key=lambda x: -x[1]):
        if count >= 2 and len(patterns) < 5:
            patterns.append(f"Resolution pattern: {keyword} ({count}/{len(tickets)} tickets)")
    
    return patterns[:5]  # Return top 5 patterns