"""
Attachment Analyzer Tool - PDF/DOCX/XLSX Analysis with Gemini
Extracts model numbers, part numbers, and key information from attachments
"""

import logging
from typing import Dict, Any, List
from langchain.tools import tool

from app.utils.attachment_processor import process_attachment
from app.clients.llm_client import get_llm_client

logger = logging.getLogger(__name__)


@tool
def attachment_analyzer_tool(
    attachments: List[Dict[str, Any]],
    analysis_focus: str = "general"
) -> Dict[str, Any]:
    """
    Analyze PDF/DOCX/XLSX attachments to extract product information.
    
    Use this tool when:
    - Customer attached invoices, receipts, or packing slips
    - Need to extract model numbers from documents
    - Invoice shows what products customer ordered
    - Document contains serial numbers or part numbers
    
    Args:
        attachments: List of attachment dicts with keys 'attachment_url', 'name', 'content_type'
        analysis_focus: What to focus on - "general" | "model_numbers" | "invoice" | 
                       "missing_parts" | "warranty"
    
    Returns:
        {
            "success": bool,
            "extracted_info": {
                "model_numbers": [str],
                "part_numbers": [str],
                "order_number": str,
                "serial_numbers": [str],
                "product_names": [str],
                "quantities": {product: int},
                "dates": {type: str},
                "key_entities": [str]
            },
            "document_summaries": [
                {
                    "filename": str,
                    "type": "invoice" | "manual" | "receipt" | "general",
                    "summary": str,
                    "extracted_text": str
                }
            ],
            "analysis": str,  # Gemini's analysis of the documents
            "count": int,
            "message": str
        }
    """
    logger.info(f"[ATTACHMENT_ANALYZER] Analyzing {len(attachments)} attachment(s), Focus: {analysis_focus}")
    
    if not attachments:
        return {
            "success": False,
            "extracted_info": {},
            "document_summaries": [],
            "analysis": "",
            "count": 0,
            "message": "No attachments to analyze"
        }
    
    try:
        document_summaries = []
        all_extracted_text = []
        
        # Process each attachment
        for att in attachments:
            # Skip images (handled by vision_search_tool)
            content_type = att.get("content_type", "").lower()
            if content_type.startswith("image/"):
                continue
            
            # Process document
            result = process_attachment(att)
            
            if result and result.content:
                doc_summary = {
                    "filename": result.filename,
                    "type": result.file_type,
                    "summary": result.content[:500],  # First 500 chars
                    "extracted_text": result.content,
                    "page_count": result.page_count,
                    "size_bytes": result.size_bytes
                }
                document_summaries.append(doc_summary)
                all_extracted_text.append(f"--- {result.filename} ---\n{result.content}")
        
        if not document_summaries:
            return {
                "success": False,
                "extracted_info": {},
                "document_summaries": [],
                "analysis": "",
                "count": 0,
                "message": "Could not extract text from any attachments"
            }
        
        # Combine all text for analysis
        combined_text = "\n\n".join(all_extracted_text)
        
        # Use Gemini to extract structured information
        extracted_info = _gemini_extract_entities(
            text=combined_text,
            focus=analysis_focus
        )
        
        # Generate analysis summary
        analysis = _gemini_analyze_documents(
            text=combined_text,
            focus=analysis_focus,
            extracted_info=extracted_info
        )
        
        logger.info(f"[ATTACHMENT_ANALYZER] Extracted: {extracted_info}")
        
        return {
            "success": True,
            "extracted_info": extracted_info,
            "document_summaries": document_summaries,
            "analysis": analysis,
            "count": len(document_summaries),
            "message": f"Analyzed {len(document_summaries)} document(s)"
        }
        
    except Exception as e:
        logger.error(f"[ATTACHMENT_ANALYZER] Error: {e}", exc_info=True)
        return {
            "success": False,
            "extracted_info": {},
            "document_summaries": [],
            "analysis": "",
            "count": 0,
            "message": f"Attachment analysis failed: {str(e)}"
        }


def _gemini_extract_entities(text: str, focus: str) -> Dict[str, Any]:
    """Use Gemini to extract structured entities from text"""
    
    system_prompt = f"""You are an expert at extracting product information from documents.

Extract the following information from the provided text:
- Model numbers (e.g., HS6270MB, F2580CP, D4500BN)
- Part numbers (e.g., #12345, PART-ABC-123)
- Order numbers (e.g., Order #12345, PO#50218)
- Serial numbers
- Product names and descriptions
- Quantities (how many of each product)
- Important dates (order date, delivery date, warranty expiration)
- Any other relevant product identifiers

Focus: {focus}

Respond ONLY with valid JSON in this exact format:
{{
    "model_numbers": ["HS6270MB", ...],
    "part_numbers": ["#12345", ...],
    "order_number": "12345" or null,
    "serial_numbers": ["SN123", ...],
    "product_names": ["Shower Head", ...],
    "quantities": {{"Shower Head": 2}},
    "dates": {{"order_date": "2024-01-15", "delivery_date": "2024-01-20"}},
    "key_entities": ["other important info"]
}}"""
    
    user_prompt = f"Document text:\n\n{text[:8000]}"  # Limit to 8000 chars
    
    try:
        llm = get_llm_client()
        response = llm.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format="json",
            temperature=0.1
        )
        
        if isinstance(response, dict):
            return response
        return {}
        
    except Exception as e:
        logger.error(f"Gemini extraction failed: {e}")
        return {}


def _gemini_analyze_documents(
    text: str,
    focus: str,
    extracted_info: Dict[str, Any]
) -> str:
    """Use Gemini to generate analysis summary"""
    
    system_prompt = f"""You are analyzing customer-provided documents.

Based on the extracted information and document content, provide a concise analysis (2-3 sentences) covering:
- What type of document(s) this is (invoice, manual, receipt, etc.)
- Key products/model numbers identified
- What the customer likely needs help with
- Any important context (order details, warranty info, etc.)

Focus: {focus}"""
    
    user_prompt = f"""Extracted Info:
{extracted_info}

Document text:
{text[:5000]}"""  # Limit to 5000 chars
    
    try:
        llm = get_llm_client()
        response = llm.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3
        )
        
        return str(response) if response else "No analysis available"
        
    except Exception as e:
        logger.error(f"Gemini analysis failed: {e}")
        return "Analysis unavailable"
