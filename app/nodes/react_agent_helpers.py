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
from app.tools.finish import finish_tool
from app.tools.attachment_analyzer import attachment_analyzer_tool
from app.tools.attachment_classifier_tool import attachment_type_classifier_tool
from app.tools.multimodal_document_analyzer import multimodal_document_analyzer_tool
from app.tools.ocr_image_analyzer import ocr_image_analyzer_tool

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

        if tool_results.get("attachment_classification"):
            ac = tool_results["attachment_classification"]
            if ac.get("success"):
                types = [f"{a['name']}: {a['detected_type']}" for a in ac.get("attachments", [])]
                context_parts.append(f"✓ Attachment Classification: {', '.join(types)}")

        if tool_results.get("multimodal_doc_analysis"):
            mda = tool_results["multimodal_doc_analysis"]
            if mda.get("success"):
                context_parts.append(f"✓ Multimodal Doc Analysis: {mda.get('count', 0)} document(s) processed")
                for doc in mda.get("documents", [])[:2]:
                    context_parts.append(f"  {doc.get('filename')}: {list(doc.get('extracted_info', {}).keys())}")

        if tool_results.get("ocr_image_analysis"):
            ocr = tool_results["ocr_image_analysis"]
            if ocr.get("success"):
                context_parts.append(f"✓ OCR Image Analysis: {len(ocr.get('results', []))} image(s) processed")
                for res in ocr.get("results", [])[:2]:
                    context_parts.append(f"  {res.get('image_url')}: {res.get('text', '')[:60]}...")
    
    # Add urgency if approaching limit - make it VERY prominent
    if iteration_num >= max_iterations - 2:
        context_parts.append(f"\n\n{'='*60}")
        context_parts.append(f"🛑 CRITICAL URGENCY ALERT 🛑")
        context_parts.append(f"{'='*60}")
        context_parts.append(f"⚠️ Only {max_iterations - iteration_num} iteration(s) remaining!")
        context_parts.append(f"⚠️ You MUST call finish_tool NOW with whatever information you have!")
        context_parts.append(f"⚠️ Do NOT attempt any more searches - you're out of time!")
        context_parts.append(f"{'='*60}\n")
    elif iteration_num >= max_iterations - 3:
        context_parts.append(f"\n⚠️ WARNING: Only {max_iterations - iteration_num} iterations remaining!")
        context_parts.append("⚠️ Consider calling finish_tool soon with your current findings.")
    
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
        def _run_tool(tool, kwargs):
            """
            Standard execution wrapper for LangChain Tools.
            All tools in LC v1 expect a single argument called `tool_input`.
            """
            # If the tool expects a single dict/string input, pass as tool_input
            if "tool_input" not in kwargs:
                # Wrap entire kwargs as tool_input
                return tool.run(tool_input=kwargs)
            
            # If caller already prepared "tool_input", pass through
            return tool.run(**kwargs)

        # ------------------------------------------------------
        # AUTO-DETECT MODEL NUMBERS FOR PRODUCT SEARCH
        # ------------------------------------------------------
        def _looks_like_model_number(text: Any) -> bool:
            import re
            return isinstance(text, str) and bool(re.match(r"^[A-Za-z0-9\-\.]{4,25}$", text))

        if action == "product_search_tool":
            # Make sure action_input is a dict we can safely mutate
            action_input = dict(action_input or {})
            # If LLM used `query` instead of `model_number`, and it looks like a model number → fix it
            if "model_number" not in action_input:
                q = action_input.get("query")
                if _looks_like_model_number(q):
                    action_input["model_number"] = q
                    action_input.pop("query", None)

        # Map action to tool function
        if action == "product_search_tool":
            output = _run_tool(product_search_tool, action_input or {})
            tool_results["product_search"] = output
            if output.get("success"):
                count = output.get("count", 0)
                obs = f"Found {count} product(s). "
                if output.get("products"):
                    top = output["products"][0]
                    obs += f"Top match: {top.get('model_no')} - {top.get('product_title')} (score: {top.get('similarity_score')}%)"
                return output, obs
            else:
                return output, f"No products found: {output.get('message')}"

        elif action == "document_search_tool":
            action_input = dict(action_input or {})
            if identified_product and not action_input.get("product_context"):
                model = identified_product.get("model")
                name = identified_product.get("name")
                if model or name:
                    action_input["product_context"] = model or name
            output = _run_tool(document_search_tool, action_input)
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
            action_input = dict(action_input or {})
            action_input["image_urls"] = ticket_images
            output = _run_tool(vision_search_tool, action_input)
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
            output = _run_tool(past_tickets_search_tool, action_input or {})
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
            action_input = dict(action_input or {})
            action_input["attachments"] = attachments
            output = _run_tool(attachment_analyzer_tool, action_input)
            tool_results["attachment_analysis"] = output
            if output.get("success"):
                extracted = output.get("extracted_info", {})
                models = extracted.get("model_numbers", [])
                obs = f"Analyzed {output.get('count', 0)} attachment(s). "
                if models:
                    obs += f"Model numbers: {', '.join(models[:5])}"
                else:
                    obs += "No model numbers extracted."
                return output, obs
            else:
                return output, f"Attachment analysis failed: {output.get('message')}"

        elif action == "attachment_type_classifier_tool":
            action_input = dict(action_input or {})
            action_input["attachments"] = attachments
            output = _run_tool(attachment_type_classifier_tool, action_input)
            tool_results["attachment_classification"] = output
            if output.get("success"):
                obs = f"Attachment types classified: {[a['detected_type'] for a in output.get('attachments', [])]}"
                return output, obs
            else:
                return output, f"Attachment type classification failed: {output.get('message')}"

        elif action == "multimodal_document_analyzer_tool":
            action_input = dict(action_input or {})
            action_input["attachments"] = attachments
            output = _run_tool(multimodal_document_analyzer_tool, action_input)
            tool_results["multimodal_doc_analysis"] = output
            if output.get("success"):
                obs = f"Multimodal document analysis complete: {output.get('count', 0)} document(s) processed"
                return output, obs
            else:
                return output, f"Multimodal document analysis failed: {output.get('message')}"

        elif action == "ocr_image_analyzer_tool":
            action_input = dict(action_input or {})
            # Accepts image_urls list
            if not action_input.get("image_urls"):
                action_input["image_urls"] = ticket_images
            output = _run_tool(ocr_image_analyzer_tool, action_input)
            tool_results["ocr_image_analysis"] = output
            if output.get("success"):
                obs = f"OCR image analysis complete: {len(output.get('results', []))} image(s) processed"
                return output, obs
            else:
                return output, f"OCR image analysis failed: {output.get('message')}"

        elif action == "finish_tool":
            output = _run_tool(finish_tool, action_input or {})
            obs = f"Finished. {output.get('summary', '')}"
            return output, obs

        else:
            obs = f"Unknown tool: {action}"
            return {"error": obs, "success": False}, obs

    except Exception as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)
        obs = f"Tool execution failed: {str(e)}"
        return {"error": obs, "success": False}, obs


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
    
    # Deduplicate documents by title (case-insensitive) - improvement from improvements.md
    seen_titles = set()
    unique_docs = []
    for doc in relevant_documents:
        title = doc.get("title", "").lower()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_docs.append(doc)
        elif not title:  # Allow docs without titles
            unique_docs.append(doc)
    relevant_documents = unique_docs
    
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
    
    # Surface Gemini answer prominently FIRST (improvement from improvements.md)
    if gemini_answer:
        context_sections.append("### 🎯 DIRECT ANSWER FROM DOCUMENTATION")
        context_sections.append(str(gemini_answer)[:1000])  # Increased from 800
        context_sections.append("")  # Blank line for readability
    
    # Add document context (increased from top 5 to top 10 - improvement from improvements.md)
    if text_retrieval_results:
        context_sections.append("### PRODUCT DOCUMENTATION")
        for i, hit in enumerate(text_retrieval_results[:10], 1):  # Increased from 5 to 10
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
    if not multimodal_context or len(multimodal_context.strip()) < 50:
        logger.warning("Multimodal context is empty or too short!")
        multimodal_context = "No context gathered. Agent did not retrieve sufficient information."
    
    # Build source_documents for citations (increased from 5 to 10 - improvement from improvements.md)
    source_documents = []
    for i, doc in enumerate(relevant_documents[:10]):  # Increased from 5 to 10
        source_documents.append({
            "rank": i + 1,
            "title": doc.get("title", "Unknown"),
            "content_preview": str(doc.get("content_preview", ""))[:500],
            "relevance_score": doc.get("relevance_score", 0),
            "source_type": "gemini_file_search",
            "uri": doc.get("uri", "")  # Include URI if available
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
