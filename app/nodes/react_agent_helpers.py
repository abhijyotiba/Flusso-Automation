"""
ReACT Agent Helper Functions
Context building, tool execution, legacy field population
"""

import logging
from typing import Dict, Any, List, Tuple

from app.tools.product_search import product_search_tool
from app.tools.document_search import document_search_tool
from app.tools.vision_search import vision_search_tool
from app.tools.past_tickets import past_tickets_search_tool
from app.tools.attachment_analyzer import attachment_analyzer_tool
from app.tools.finish import finish_tool

logger = logging.getLogger(__name__)


def _build_agent_context(
    ticket_subject: str,
    ticket_text: str,
    ticket_images: List[str],
    attachments: List[Dict],
    iterations: List[Dict],
    tool_results: Dict[str, Any],
    iteration_num: int,
    max_iterations: int
) -> str:
    """Build context for ReACT agent with ticket info and history"""
    
    context_parts = [
        f"═══ ITERATION {iteration_num}/{max_iterations} ═══\n",
        f"CUSTOMER TICKET:",
        f"Subject: {ticket_subject}",
        f"\nDescription:\n{ticket_text[:2000]}",  # Limit to 2000 chars
    ]
    
    if ticket_images:
        context_parts.append(f"\n📷 Images Attached: {len(ticket_images)} image(s)")
        context_parts.append(f"Image URLs: {ticket_images}")
    
    if attachments:
        context_parts.append(f"\n📎 Documents Attached: {len(attachments)} file(s)")
        att_info = []
        for att in attachments[:5]:  # Show first 5
            att_info.append(f"  - {att.get('filename', 'Unknown')} ({att.get('type', 'unknown')})")
        context_parts.append("\n".join(att_info))
    
    # Show what tools have been used
    if iterations:
        context_parts.append(f"\n\n═══ PREVIOUS ACTIONS ═══")
        for it in iterations[-5:]:  # Show last 5
            context_parts.append(f"\nIteration {it['iteration']}:")
            context_parts.append(f"  Thought: {it['thought'][:150]}")
            context_parts.append(f"  Action: {it['action']}")
            context_parts.append(f"  Result: {it['observation'][:200]}")
    
    # Show accumulated results
    context_parts.append(f"\n\n═══ CURRENT STATE ═══")
    
    if tool_results["product_search"]:
        prod = tool_results["product_search"]
        if prod.get("success"):
            context_parts.append(f"✓ Product Search: Found {prod.get('count', 0)} product(s)")
            if prod.get("products"):
                top = prod["products"][0]
                context_parts.append(f"  Top match: {top.get('model_no')} - {top.get('product_title')}")
    
    if tool_results["document_search"]:
        docs = tool_results["document_search"]
        if docs.get("success"):
            context_parts.append(f"✓ Document Search: Found {docs.get('count', 0)} document(s)")
    
    if tool_results["vision_search"]:
        vis = tool_results["vision_search"]
        if vis.get("success"):
            context_parts.append(f"✓ Vision Search: Quality={vis.get('match_quality')}, {vis.get('count', 0)} match(es)")
    
    if tool_results["past_tickets"]:
        past = tool_results["past_tickets"]
        if past.get("success"):
            context_parts.append(f"✓ Past Tickets: Found {past.get('count', 0)} similar ticket(s)")
    
    if tool_results["attachment_analysis"]:
        att_analysis = tool_results["attachment_analysis"]
        if att_analysis.get("success"):
            extracted = att_analysis.get("extracted_info", {})
            models = extracted.get("model_numbers", [])
            if models:
                context_parts.append(f"✓ Attachment Analysis: Model numbers extracted: {models}")
    
    # Add urgency if approaching limit
    if iteration_num >= max_iterations - 2:
        context_parts.append(f"\n⚠️ WARNING: Only {max_iterations - iteration_num} iterations remaining!")
        context_parts.append("You MUST call finish_tool in the next 1-2 iterations with whatever you've gathered.")
    
    return "\n".join(context_parts)


def _execute_tool(
    action: str,
    action_input: Dict[str, Any],
    ticket_images: List[str],
    attachments: List[Dict],
    tool_results: Dict[str, Any]
) -> Tuple[Dict[str, Any], str]:
    """
    Execute the chosen tool and return (output, observation_summary)
    """
    
    try:
        # Map action to tool function
        if action == "product_search_tool":
            output = product_search_tool.invoke(action_input)
            tool_results["product_search"] = output
            
            if output.get("success"):
                count = output.get("count", 0)
                obs = f"Found {count} product(s). "
                if output.get("products"):
                    top = output["products"][0]
                    obs += f"Top match: {top['model_no']} - {top['product_title']} (score: {top['similarity_score']}%)"
                return output, obs
            else:
                return output, f"No products found: {output.get('message')}"
        
        elif action == "document_search_tool":
            output = document_search_tool.invoke(action_input)
            tool_results["document_search"] = output
            
            if output.get("success"):
                count = output.get("count", 0)
                obs = f"Found {count} document(s). "
                if output.get("documents"):
                    titles = [d.get("title", "Untitled") for d in output["documents"][:3]]
                    obs += f"Top docs: {', '.join(titles)}"
                return output, obs
            else:
                return output, f"No documents found: {output.get('message')}"
        
        elif action == "vision_search_tool":
            # Inject ticket_images into action_input
            action_input["image_urls"] = ticket_images
            output = vision_search_tool.invoke(action_input)
            tool_results["vision_search"] = output
            
            if output.get("success"):
                quality = output.get("match_quality")
                count = output.get("count", 0)
                obs = f"Vision match quality: {quality}. Found {count} match(es). "
                obs += output.get("reasoning", "")
                return output, obs
            else:
                return output, f"Vision search failed: {output.get('message')}"
        
        elif action == "past_tickets_search_tool":
            output = past_tickets_search_tool.invoke(action_input)
            tool_results["past_tickets"] = output
            
            if output.get("success"):
                count = output.get("count", 0)
                obs = f"Found {count} similar past ticket(s). "
                patterns = output.get("common_patterns", [])
                if patterns:
                    obs += f"Patterns: {'; '.join(patterns)}"
                return output, obs
            else:
                return output, f"No past tickets found: {output.get('message')}"
        
        elif action == "attachment_analyzer_tool":
            # Inject attachments into action_input
            action_input["attachments"] = attachments
            output = attachment_analyzer_tool.invoke(action_input)
            tool_results["attachment_analysis"] = output
            
            if output.get("success"):
                extracted = output.get("extracted_info", {})
                models = extracted.get("model_numbers", [])
                obs = f"Analyzed {output.get('count', 0)} document(s). "
                if models:
                    obs += f"Model numbers: {', '.join(models[:5])}"
                else:
                    obs += "No model numbers extracted."
                return output, obs
            else:
                return output, f"Attachment analysis failed: {output.get('message')}"
        
        elif action == "finish_tool":
            output = finish_tool.invoke(action_input)
            obs = f"Finished. {output.get('summary', '')}"
            return output, obs
        
        else:
            obs = f"Unknown tool: {action}"
            return {"error": obs}, obs
            
    except Exception as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)
        obs = f"Tool execution failed: {str(e)}"
        return {"error": obs}, obs


def _populate_legacy_fields(
    gathered_documents: List[Dict],
    gathered_images: List,  # Can be list of strings or dicts
    gathered_past_tickets: List[Dict],
    identified_product: Dict = None,
    product_confidence: float = 0.0
) -> Dict[str, Any]:
    """
    Populate legacy RAG result fields for compatibility with existing nodes
    (draft_response, confidence_check, etc. still expect these fields)
    """
    
    # Convert gathered_documents to RetrievalHit format
    text_retrieval_results = []
    for i, doc in enumerate(gathered_documents):
        if isinstance(doc, dict):
            text_retrieval_results.append({
                "id": doc.get("id", f"doc_{i}"),
                "score": doc.get("relevance_score", 0.8),
                "metadata": {
                    "title": doc.get("title", "Unknown"),
                    "source": "gemini_file_search"
                },
                "content": doc.get("content_preview", "")
            })
        else:
            text_retrieval_results.append({
                "id": str(doc),
                "score": 0.8,
                "metadata": {"title": str(doc), "source": "react_agent"},
                "content": ""
            })
    
    # Convert gathered_images to RetrievalHit format
    # Handle both string URLs and dict formats
    image_retrieval_results = []
    for i, img in enumerate(gathered_images):
        if isinstance(img, str):
            img_url = img
        elif isinstance(img, dict):
            img_url = img.get("url") or img.get("image_url") or img.get("src") or ""
        else:
            img_url = str(img)
        
        if img_url:
            image_retrieval_results.append({
                "id": f"img_{i}",
                "score": 0.9,  # Assume high quality since React agent validated
                "metadata": {
                    "image_url": img_url,
                    "source": "react_vision"
                },
                "content": f"Product image {i+1}"
            })
    
    # Convert gathered_past_tickets to RetrievalHit format
    past_ticket_results = []
    for i, ticket in enumerate(gathered_past_tickets):
        if isinstance(ticket, dict):
            similarity = ticket.get("similarity_score", 0)
            # Handle percentage vs decimal
            if isinstance(similarity, (int, float)) and similarity > 1:
                similarity = similarity / 100.0
            past_ticket_results.append({
                "id": f"ticket_{ticket.get('ticket_id', i)}",
                "score": similarity,
                "metadata": {
                    "ticket_id": ticket.get("ticket_id"),
                    "subject": ticket.get("subject"),
                    "resolution_type": ticket.get("resolution_type"),
                    "source": "past_tickets"
                },
                "content": ticket.get("resolution_summary", "")
            })
        else:
            past_ticket_results.append({
                "id": str(ticket),
                "score": 0.5,
                "metadata": {"ticket_id": str(ticket), "source": "past_tickets"},
                "content": ""
            })
    
    # Build source_documents for citations
    source_documents = []
    for i, doc in enumerate(gathered_documents[:5]):
        if isinstance(doc, dict):
            source_documents.append({
                "rank": i + 1,
                "title": doc.get("title", "Unknown"),
                "content_preview": str(doc.get("content_preview", ""))[:500],
                "relevance_score": doc.get("relevance_score", 0),
                "source_type": "gemini_file_search"
            })
    
    # Build source_products for citations from identified_product
    source_products = []
    if identified_product and isinstance(identified_product, dict):
        source_products.append({
            "rank": 1,
            "model_no": identified_product.get("model", "Unknown"),
            "product_title": identified_product.get("name", "Unknown"),
            "category": identified_product.get("category", "Unknown"),
            "similarity_score": int(product_confidence * 100),
            "source_type": "react_agent"
        })
    
    # Build source_tickets for citations
    source_tickets = []
    for i, ticket in enumerate(gathered_past_tickets[:5]):
        if isinstance(ticket, dict):
            source_tickets.append({
                "rank": i + 1,
                "ticket_id": ticket.get("ticket_id"),
                "subject": ticket.get("subject"),
                "resolution_type": ticket.get("resolution_type"),
                "resolution_summary": str(ticket.get("resolution_summary", ""))[:200],
                "similarity_score": ticket.get("similarity_score", 0),
                "source_type": "past_tickets"
            })
    
    # Determine if we have enough information
    has_docs = len(text_retrieval_results) > 0
    has_images = len(image_retrieval_results) > 0
    has_product = identified_product is not None
    enough_info = has_docs or has_images or has_product
    
    return {
        "text_retrieval_results": text_retrieval_results,
        "image_retrieval_results": image_retrieval_results,
        "past_ticket_results": past_ticket_results,
        "source_documents": source_documents,
        "source_products": source_products,
        "source_tickets": source_tickets,
        "enough_information": enough_info,
        "product_match_confidence": product_confidence,
        "overall_confidence": product_confidence
    }
