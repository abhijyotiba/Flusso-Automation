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
    
    Use this tool when you need:
    - Installation instructions
    - Product specifications
    - Troubleshooting guides
    - Warranty information
    - Maintenance procedures
    - Parts lists
    - Technical diagrams
    
    Args:
        query: What information you're looking for (e.g., "installation instructions", 
               "leak repair", "warranty period")
        product_context: Product model/name to provide context (e.g., "HS6270MB shower head")
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
        client = get_gemini_client()
        
        # Build context-aware query
        if product_context:
            search_query = f"""
Product: {product_context}

Customer needs help with: {query}

Find relevant documentation from product manuals, installation guides, FAQs, or warranty information.
"""
        else:
            search_query = f"""
Customer needs help with: {query}

Find relevant documentation from product manuals, installation guides, FAQs, or warranty information.
"""
        
        # Execute Gemini File Search with sources
        result = client.search_files_with_sources(
            query=search_query,
            top_k=top_k,
            system_instruction="""You are a product documentation search assistant.
Your job is to find the most relevant documentation to help answer the customer's question.
Be thorough and cite multiple relevant sources."""
        )
        
        hits = result.get('hits', [])
        gemini_answer = result.get('gemini_answer', '')
        source_documents = result.get('source_documents', [])
        
        if not source_documents:
            return {
                "success": False,
                "documents": [],
                "gemini_answer": gemini_answer,
                "count": 0,
                "message": "No relevant documentation found"
            }
        
        # Format documents for ReACT agent
        documents = []
        for doc in source_documents[:top_k]:
            documents.append({
                "title": doc.get("title", "Unknown Document"),
                "content_preview": doc.get("content_preview", ""),
                "relevance_score": doc.get("relevance_score", 0),
                "source_type": _infer_document_type(doc.get("title", "")),
                "rank": doc.get("rank", 0)
            })
        
        logger.info(f"[DOCUMENT_SEARCH] Found {len(documents)} relevant document(s)")
        
        return {
            "success": True,
            "documents": documents,
            "gemini_answer": gemini_answer,
            "count": len(documents),
            "message": f"Found {len(documents)} relevant document(s)"
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
