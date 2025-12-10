"""
Multimodal Document Analyzer Tool - FIXED
Uses Gemini to extract structured data from PDFs and documents via File API
"""

import logging
import os
import json
import requests
import tempfile
from typing import Dict, Any, List
from langchain.tools import tool
from google import genai
from google.genai import types
from requests.auth import HTTPBasicAuth

# Import settings globally
from app.config.settings import settings

logger = logging.getLogger(__name__)


def _download_attachment(url: str, name: str) -> str:
    """Download attachment to temp file and return path"""
    try:
        # Use Freshdesk API key from settings
        auth = HTTPBasicAuth(settings.freshdesk_api_key, "X")
        
        logger.info(f"[DOC_ANALYZER] Downloading: {name}")
        response = requests.get(url, auth=auth, timeout=30)
        response.raise_for_status()
        
        # Create temp file with proper extension
        # Default to .pdf if no extension found
        suffix = "." + name.split(".")[-1] if "." in name else ".pdf"
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        
        with open(path, "wb") as f:
            f.write(response.content)
        
        logger.info(f"[DOC_ANALYZER] Downloaded to: {path}")
        return path
        
    except Exception as e:
        logger.error(f"[DOC_ANALYZER] Download failed for {name}: {e}")
        raise


@tool
def multimodal_document_analyzer_tool(
    attachments: List[Dict[str, Any]],
    focus: str = "model_numbers"
) -> Dict[str, Any]:
    """
    Analyze documents using Gemini Vision to extract structured information.
    
    Focuses on extracting:
    - Product model numbers (e.g., "100.1170", "HS6270MB")
    - Part numbers (e.g., "160.1168-9862")
    - Order/PO numbers
    - Product names and descriptions
    - Quantities and prices
    - Dates
    
    Args:
        attachments: List of attachment dicts with 'attachment_url' and 'name'
        focus: What to extract - "model_numbers", "order_info", or "general"
    
    Returns:
        Dict containing extracted information and document counts.
    """
    logger.info(f"[DOC_ANALYZER] Processing {len(attachments)} attachment(s)")
    
    if not attachments:
        return {
            "success": False,
            "documents": [],
            "count": 0,
            "message": "No attachments provided"
        }
    
    try:
        # Initialize Gemini client using settings
        if not settings.gemini_api_key:
             return {"success": False, "message": "Missing GEMINI_API_KEY"}
             
        client = genai.Client(api_key=settings.gemini_api_key)
        
        documents = []
        temp_files = []  # Track temp files for cleanup
        
        for att in attachments:
            url = att.get("attachment_url")
            name = att.get("name", "unknown")
            
            if not url:
                logger.warning(f"[DOC_ANALYZER] No URL for attachment: {name}")
                continue
            
            try:
                # 1. Download to temp file
                local_path = _download_attachment(url, name)
                temp_files.append(local_path)
                
                # 2. Upload to Gemini Files API
                logger.info(f"[DOC_ANALYZER] Uploading {name} to Gemini")
                file_obj = client.files.upload(file=local_path)
                
                # 3. Build extraction prompt
                if focus == "model_numbers":
                    extraction_prompt = """Extract ALL product model numbers and part numbers from this document.

Focus on patterns like:
- Model numbers: 100.1170, HS6270MB, F2580CP, 160.1168-9862
- Part numbers: Any alphanumeric codes that look like product identifiers
- Item codes or SKUs

Return JSON:
{
  "model_numbers": ["100.1170", "HS6270MB", ...],
  "part_numbers": ["160.1168-9862", ...],
  "product_names": ["Water Filler", "Shower Head", ...],
  "order_numbers": ["PO-12345", ...],
  "quantities": {"100.1170": 2},
  "dates": {"order_date": "2024-01-15"},
  "key_entities": ["any other important info"]
}

IMPORTANT: Extract EVERY model/part number you see, even if it appears multiple times."""
                
                elif focus == "order_info":
                    extraction_prompt = """Extract order/invoice information from this document.

Focus on:
- Order/PO numbers
- Product model numbers and names
- Quantities ordered
- Prices
- Order date
- Customer information

Return JSON with all extracted fields."""
                
                else:  # general
                    extraction_prompt = """Extract all structured information from this document.

Return JSON with:
- model_numbers: list of product models
- part_numbers: list of part numbers
- order_numbers: list of order/PO numbers
- product_names: list of product names
- quantities: dict mapping products to quantities
- dates: dict with relevant dates
- key_entities: list of other important information"""
                
                # 4. Call Gemini for extraction
                logger.info(f"[DOC_ANALYZER] Analyzing {name} with model: {settings.llm_model}")
                
                # Use strict structure to avoid Pydantic validation errors
                response = client.models.generate_content(
                    model=settings.llm_model,  # Use configured model (e.g., gemini-2.5-flash)
                    contents=[
                        types.Content(
                            parts=[
                                types.Part(text=extraction_prompt),
                                types.Part(
                                    file_data=types.FileData(
                                        file_uri=file_obj.uri,
                                        mime_type=file_obj.mime_type
                                    )
                                )
                            ]
                        )
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1  # Low temp for accurate extraction
                    )
                )
                
                # 5. Parse JSON response
                extracted_text = response.text if response.text else ""
                
                if extracted_text:
                    try:
                        extracted_info = json.loads(extracted_text)
                    except json.JSONDecodeError as e:
                        logger.error(f"[DOC_ANALYZER] JSON parse error for {name}: {e}")
                        extracted_info = {"error": "Failed to parse JSON", "raw": extracted_text}
                else:
                    extracted_info = {"error": "No response text from Gemini"}
                
                documents.append({
                    "filename": name,
                    "extracted_info": extracted_info
                })
                
                logger.info(f"[DOC_ANALYZER] ✓ Analyzed {name}")
                
            except Exception as e:
                logger.error(f"[DOC_ANALYZER] Failed to process {name}: {e}", exc_info=True)
                documents.append({
                    "filename": name,
                    "extracted_info": {"error": str(e)}
                })
        
        # 6. Cleanup temp files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"[DOC_ANALYZER] Failed to delete temp file {temp_file}: {e}")
        
        logger.info(f"[DOC_ANALYZER] ✅ Completed: {len(documents)} document(s) processed")
        
        return {
            "success": True,
            "documents": documents,
            "count": len(documents),
            "message": f"Successfully analyzed {len(documents)} document(s)"
        }
        
    except Exception as e:
        logger.error(f"[DOC_ANALYZER] Critical error: {e}", exc_info=True)
        return {
            "success": False,
            "documents": [],
            "count": 0,
            "message": f"Document analysis failed: {str(e)}"
        }