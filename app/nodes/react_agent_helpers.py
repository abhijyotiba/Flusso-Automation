"""
ReACT Agent Helper Functions - FIXED VERSION
Context building, tool execution, legacy field population
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

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
    tool_results: Dict[str, Any],
    identified_product: Optional[Dict[str, Any]] = None
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
            # Make a defensive copy so we never mutate caller input
            action_input = dict(action_input or {})
            # If we already know the product but the LLM forgot to add context,
            # inject it to dramatically improve search quality
            if identified_product and not action_input.get("product_context"):
                model = identified_product.get("model")
                name = identified_product.get("name")
                if model or name:
                    action_input["product_context"] = model or name

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
    gathered_images: List,
    gathered_past_tickets: List[Dict],
    identified_product: Dict = None,
    product_confidence: float = 0.0,
    gemini_answer: str = ""
) -> Dict[str, Any]:
    """
    Populate legacy RAG result fields for compatibility with existing nodes.
    CRITICAL FIX: Properly formats all fields and builds multimodal_context string.
    """
    
    # Normalize and validate inputs
    product_details = identified_product or {}
    relevant_documents = _normalize_documents(gathered_documents)
    relevant_images = _normalize_images(gathered_images)
    past_tickets = _normalize_tickets(gathered_past_tickets)
    
    # Convert to RetrievalHit format for legacy nodes
    text_retrieval_results = []
    for i, doc in enumerate(relevant_documents):
        text_retrieval_results.append({
            "id": doc.get("id", f"doc_{i}"),
            "score": doc.get("relevance_score", 0.8),
            "metadata": {
                "title": doc.get("title", "Unknown"),
                "source": "gemini_file_search"
            },
            "content": doc.get("content_preview", doc.get("title", ""))
        })
    
    # Convert images to RetrievalHit format
    image_retrieval_results = []
    for i, img_url in enumerate(relevant_images):
        if img_url:
            image_retrieval_results.append({
                "id": f"img_{i}",
                "score": 0.9,
                "metadata": {
                    "image_url": img_url,
                    "source": "react_vision"
                },
                "content": f"Product image {i+1}"
            })
    
    # Convert past tickets to RetrievalHit format
    past_ticket_results = []
    for i, ticket in enumerate(past_tickets):
        similarity = ticket.get("similarity_score", 0)
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
    
    # ========== CRITICAL FIX: Build multimodal_context string ==========
    # This is what downstream nodes (context_builder, orchestration, draft_response) expect!
    context_sections = []
    
    # Add document context
    if text_retrieval_results:
        context_sections.append("### PRODUCT DOCUMENTATION")
        for i, hit in enumerate(text_retrieval_results[:5], 1):
            title = hit.get("metadata", {}).get("title", f"Document {i}")
            content = hit.get("content", "")[:500]
            score = hit.get("score", 0.0)
            context_sections.append(f"{i}. **{title}** (score: {score:.2f})\n{content}\n")
    
    # Add product/vision context
    if image_retrieval_results and identified_product:
        context_sections.append("\n### PRODUCT MATCHES (VISUAL)")
        model = identified_product.get("model", "Unknown")
        name = identified_product.get("name", "Product")
        category = identified_product.get("category", "Unknown")
        context_sections.append(f"Identified Product: {name} (Model: {model}, Category: {category})")
        context_sections.append(f"Confidence: {product_confidence:.2%}")
    
    # If Gemini produced a grounded answer, surface it for downstream nodes
    if gemini_answer:
        context_sections.append("\n### DIRECT GEMINI ANSWER")
        context_sections.append(str(gemini_answer)[:800])

    # Add past tickets context
    if past_ticket_results:
        context_sections.append("\n### SIMILAR PAST TICKETS")
        for i, hit in enumerate(past_ticket_results[:3], 1):
            meta = hit.get("metadata", {})
            ticket_id = meta.get("ticket_id", "Unknown")
            resolution_type = meta.get("resolution_type", "N/A")
            content = hit.get("content", "")[:300]
            score = hit.get("score", 0.0)
            context_sections.append(
                f"{i}. Ticket #{ticket_id} ({resolution_type}) - Similarity: {score:.2f}\n{content}\n"
            )
    
    multimodal_context = "\n".join(context_sections) if context_sections else "No relevant context found."
    
    # Build source_documents for citations
    source_documents = []
    for i, doc in enumerate(relevant_documents[:5]):
        source_documents.append({
            "rank": i + 1,
            "title": doc.get("title", "Unknown"),
            "content_preview": str(doc.get("content_preview", ""))[:500],
            "relevance_score": doc.get("relevance_score", 0),
            "source_type": "gemini_file_search"
        })
    
    # Build source_products for citations
    source_products = []
    if identified_product:
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
    for i, ticket in enumerate(past_tickets[:5]):
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
    
    logger.info(f"[LEGACY_FIELDS] Populated: docs={len(text_retrieval_results)}, "
                f"images={len(image_retrieval_results)}, tickets={len(past_ticket_results)}, "
                f"context_len={len(multimodal_context)}, enough_info={enough_info}")
    
    return {
        "text_retrieval_results": text_retrieval_results,
        "image_retrieval_results": image_retrieval_results,
        "past_ticket_results": past_ticket_results,
        "multimodal_context": multimodal_context,  # CRITICAL: This was missing!
        "source_documents": source_documents,
        "source_products": source_products,
        "source_tickets": source_tickets,
        "gemini_answer": gemini_answer,
        "enough_information": enough_info,
        "product_match_confidence": product_confidence,
        "overall_confidence": product_confidence,
        # Set ran flags to prevent re-running RAG
        "ran_vision": True,
        "ran_text_rag": True,
        "ran_past_tickets": True
    }


def _normalize_documents(docs: List[Any]) -> List[Dict[str, Any]]:
    """Normalize documents list - handles both strings and dicts."""
    if not docs:
        return []
    
    normalized = []
    for doc in docs:
        if isinstance(doc, dict):
            normalized.append(doc)
        elif isinstance(doc, str):
            normalized.append({"id": doc, "title": doc, "content_preview": ""})
        else:
            normalized.append({"id": str(doc), "title": str(doc), "content_preview": ""})
    return normalized


def _normalize_images(images: List[Any]) -> List[str]:
    """Normalize images list - extracts URLs from dicts or keeps strings."""
    if not images:
        return []
    
    normalized = []
    for img in images:
        if isinstance(img, dict):
            url = img.get("url") or img.get("image_url") or img.get("src") or ""
            if url:
                normalized.append(url)
        elif isinstance(img, str):
            normalized.append(img)
    return normalized


def _normalize_tickets(tickets: List[Any]) -> List[Dict[str, Any]]:
    """Normalize tickets list - handles both strings and dicts."""
    if not tickets:
        return []
    
    normalized = []
    for ticket in tickets:
        if isinstance(ticket, dict):
            normalized.append(ticket)
        elif isinstance(ticket, str):
            normalized.append({"ticket_id": ticket, "subject": "Unknown", "resolution_summary": ""})
        else:
            normalized.append({"ticket_id": str(ticket), "subject": "Unknown", "resolution_summary": ""})
    return normalized