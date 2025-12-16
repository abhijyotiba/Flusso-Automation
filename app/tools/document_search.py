"""
Document Search Tool - Gemini File Search
Searches product manuals, FAQs, installation guides, warranty docs
"""

import logging
from typing import Dict, Any, Optional, List
from langchain.tools import tool

from app.clients.gemini_client import get_gemini_client

logger = logging.getLogger(__name__)


@tool
def document_search_tool(
    query: str,
    product_context: Optional[str] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Search product documentation using Gemini File Search.
    
    🔍 WHAT THIS TOOL CONTAINS:
    ✅ Product specifications and technical details
    ✅ Installation guides and manuals
    ✅ Parts lists with part numbers 
    ✅ Technical diagrams and schematics
    
    ❌ WHAT THIS TOOL DOES NOT CONTAIN:
    ❌ Customer order information
    ❌ Purchase orders or invoices
    ❌ Shipping/tracking information
    ❌ Customer account details
    
    ⚠️ CRITICAL: Do NOT use order numbers to search this tool.
    Order numbers are NOT product identifiers and will not yield relevant results.
    
    Args:
        query: What information you're looking for (e.g., "installation instructions", 
               "leak repair", "warranty period", "part number for diverter")
        product_context: Product model/name to provide context (e.g., "HS6270MB shower head" or "260.2693T")
        top_k: Number of documents to retrieve (default: 5)
    
    Returns:
        {
            "success": bool,
            "documents": [
                {
                    "title": str,
                    "content_preview": str,
                    "relevance_score": float,
                    "source_type": "manual" | "faq" | "warranty" | "guide"
                }
            ],
            "gemini_answer": str,  # Direct answer from Gemini if available
            "count": int,
            "message": str
        }
    """
    logger.info(f"[DOCUMENT_SEARCH] Query: '{query}', Product Context: {product_context}")
    
    try:
        if not query or not str(query).strip():
            return {
                "success": False,
                "documents": [],
                "gemini_answer": "",
                "count": 0,
                "message": "Query text is required for document search"
            }

        # Keep top_k within sane bounds to avoid overwhelming the API/LLM
        top_k = max(1, min(int(top_k), 15))
        clean_query = str(query).strip()

        client = get_gemini_client()
        
        # Determine the search strategy based on context
        search_type = _determine_search_type(clean_query)
        
        # Build context-aware query with improved formatting for better Gemini results
        if product_context:
            # Extract model number from product context if available
            model_number = _extract_model_number(product_context)
            
            # Build product-specific search query
            search_query = f"""
PRODUCT-SPECIFIC DOCUMENTATION SEARCH:

Target Product: {product_context}
{f"Model Number: {model_number}" if model_number else ""}
Customer Issue: {clean_query}

CRITICAL INSTRUCTIONS:
1. PRIORITIZE documents that specifically mention the product model "{model_number or product_context}"
2. Search for product manuals, specifications, installation guides, and troubleshooting docs for THIS SPECIFIC PRODUCT
3. DO NOT return general policy documents unless directly relevant to the specific issue
4. Focus on finding technical documentation, user manuals, parts lists for the identified product

DOCUMENT PRIORITY ORDER:
1. Product manual or datasheet for {model_number or product_context}
2. Technical specifications for this product model
3. Installation/assembly guide for this specific product
4. Troubleshooting guide for product-specific issues
5. Parts diagram or replacement parts list
6. Warranty/returns policy ONLY if customer is asking about warranty/replacement

Return specific, actionable information with part numbers, dimensions, and step-by-step instructions where applicable.
Cite the exact document source for each piece of information.
"""
        else:
            search_query = f"""
CUSTOMER SUPPORT DOCUMENTATION SEARCH:

Customer Issue: {clean_query}

TASK:
Find the most relevant documentation to help resolve this customer's issue.

DOCUMENT PRIORITY ORDER:
1. Installation guides if asking about setup/mounting/assembly
2. Troubleshooting guides if reporting problems/defects/leaks
3. Parts diagrams/lists if asking about missing/replacement parts
4. Warranty documentation if asking about coverage/claims
5. Product specifications if asking about dimensions/compatibility/features

Return specific, actionable information with part numbers and step-by-step instructions where applicable.
"""
        
        # Build context-aware system instruction
        if product_context:
            model_number = _extract_model_number(product_context)
            system_instruction = f"""You are a product documentation search assistant specializing in finding product-specific technical documentation.

CRITICAL: Prioritize finding documentation that specifically mentions or relates to the product model "{model_number or product_context}".

Your job is to:
1. Find product manuals, datasheets, and specifications for the specific product model
2. Locate installation guides, troubleshooting docs, and parts lists for this product
3. Only include general policy documents if the customer is specifically asking about warranty/returns
4. AVOID returning generic policy documents when product-specific documentation is needed

Be thorough and cite the exact document source for each piece of information."""
        else:
            system_instruction = """You are a product documentation search assistant.
Your job is to find the most relevant documentation to help answer the customer's question.
Be thorough and cite multiple relevant sources."""
        
        # Execute Gemini File Search with sources
        result = client.search_files_with_sources(
            query=search_query,
            top_k=top_k,
            system_instruction=system_instruction
        )
        
        hits = result.get('hits', [])
        gemini_answer = result.get('gemini_answer', '')
        source_documents = result.get('source_documents', [])
        
        if not source_documents:
            if gemini_answer:
                logger.warning("[DOCUMENT_SEARCH] No grounded sources, returning Gemini answer as fallback")
                return {
                    "success": True,
                    "documents": [{
                        "id": "gemini_answer",
                        "title": "Gemini Generated Answer",
                        "content_preview": gemini_answer[:500],
                        "relevance_score": 0.8,
                        "source_type": "gemini_generated",
                        "uri": ""
                    }],
                    "gemini_answer": gemini_answer,
                    "count": 1,
                    "message": "Returned generated answer (no grounded documents)",
                    "source_documents": [],
                    "hits": hits
                }

            return {
                "success": False,
                "documents": [],
                "gemini_answer": gemini_answer,
                "count": 0,
                "message": "No relevant documentation found"
            }
        
        # Format documents for ReACT agent
        documents = []
        # Try to map hits to titles for richer previews
        hit_lookup = {}
        for hit in hits:
            title = hit.get("metadata", {}).get("title", "")
            if title:
                hit_lookup[title.lower()] = hit

        for idx, doc in enumerate(source_documents[:top_k]):
            title = doc.get("title", "Unknown Document")
            lower_title = title.lower()
            mapped_hit = hit_lookup.get(lower_title)
            # Prefer preview text from hit.content when available
            content_preview = doc.get("content_preview", "")
            if mapped_hit and mapped_hit.get("content"):
                preview_from_hit = str(mapped_hit.get("content", ""))[:500]
                if preview_from_hit:
                    content_preview = preview_from_hit

            uri = doc.get("uri") or (mapped_hit.get("metadata", {}).get("uri") if mapped_hit else "")
            doc_id = doc.get("id") or (mapped_hit.get("id") if mapped_hit else f"doc_{idx}")

            documents.append({
                "id": doc_id,
                "title": title,
                "content_preview": content_preview,
                "relevance_score": doc.get("relevance_score", 0),
                "source_type": _infer_document_type(title),
                "rank": doc.get("rank", 0),
                "uri": uri
            })
        
        logger.info(f"[DOCUMENT_SEARCH] Found {len(documents)} relevant document(s)")
        
        return {
            "success": True,
            "documents": documents,
            "gemini_answer": gemini_answer,
            "count": len(documents),
            "message": f"Found {len(documents)} relevant document(s)",
            # Return raw sources/hits for deeper debugging or downstream enrichment
            "source_documents": source_documents,
            "hits": hits
        }
        
    except Exception as e:
        logger.error(f"[DOCUMENT_SEARCH] Error: {e}", exc_info=True)
        return {
            "success": False,
            "documents": [],
            "gemini_answer": "",
            "count": 0,
            "message": f"Document search failed: {str(e)}"
        }


def _infer_document_type(title: str) -> str:
    """Infer document type from title"""
    title_lower = title.lower()
    
    if any(kw in title_lower for kw in ["install", "assembly", "setup"]):
        return "installation_guide"
    elif any(kw in title_lower for kw in ["manual", "user guide", "instructions"]):
        return "user_manual"
    elif any(kw in title_lower for kw in ["warranty", "guarantee"]):
        return "warranty"
    elif any(kw in title_lower for kw in ["faq", "troubleshoot", "problem"]):
        return "troubleshooting"
    elif any(kw in title_lower for kw in ["parts", "components", "diagram"]):
        return "parts_list"
    elif any(kw in title_lower for kw in ["spec", "technical", "dimension"]):
        return "specifications"
    else:
        return "general_documentation"


def _extract_model_number(product_context: str) -> Optional[str]:
    """
    Extract model number from product context.
    Model numbers typically follow patterns like: 100.1050SB, ABC-123, PROD-456-XL
    """
    import re
    
    if not product_context:
        return None
    
    # Common model number patterns:
    # 1. Alphanumeric with dots: 100.1050SB
    # 2. Letters-numbers: ABC123, PROD456
    # 3. Hyphenated: ABC-123-XL
    patterns = [
        r'\b(\d{3}\.\d{4}[A-Z]{1,3})\b',  # Pattern: 100.1050SB
        r'\b([A-Z]{2,5}-\d{3,5}(?:-[A-Z]{1,3})?)\b',  # Pattern: PROD-123-XL
        r'\b([A-Z]{2,4}\d{3,6}[A-Z]{0,3})\b',  # Pattern: ABC123, PROD456XL
        r'\b(\d{2,3}-\d{3,4}[A-Z]{0,3})\b',  # Pattern: 10-1234AB
    ]
    
    for pattern in patterns:
        match = re.search(pattern, product_context, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Fallback: look for any word that looks like a model number (mix of letters and numbers)
    words = product_context.split()
    for word in words:
        # Has both letters and numbers, reasonable length
        if (any(c.isdigit() for c in word) and 
            any(c.isalpha() for c in word) and 
            4 <= len(word) <= 15):
            return word.upper()
    
    return None


def _determine_search_type(query: str) -> str:
    """
    Determine the type of search based on query content.
    This helps in prioritizing different document types.
    """
    query_lower = query.lower()
    
    # Issue type detection
    if any(kw in query_lower for kw in ["replace", "replacement", "new one", "send another"]):
        return "replacement_request"
    elif any(kw in query_lower for kw in ["install", "setup", "assemble", "mount"]):
        return "installation"
    elif any(kw in query_lower for kw in ["broken", "damage", "defect", "leak", "not working"]):
        return "troubleshooting"
    elif any(kw in query_lower for kw in ["warranty", "guarantee", "coverage"]):
        return "warranty"
    elif any(kw in query_lower for kw in ["missing", "parts", "component"]):
        return "parts_inquiry"
    elif any(kw in query_lower for kw in ["dimension", "size", "compatible", "fit"]):
        return "specifications"
    else:
        return "general"
