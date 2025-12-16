"""
Multimodal Document Analyzer Tool - INTELLIGENT VERSION
Classifies documents first, then analyzes contextually based on document type.
Uses Gemini to extract structured data from PDFs and documents via File API.

LARGE DOCUMENT HANDLING:
- Documents up to 10MB: Full analysis
- Documents 10-20MB: Analysis with text truncation warning
- Documents > 20MB: Rejected (Gemini file limit)
- Very long text (>100K chars): Smart truncation with ellipsis
"""

import logging
import os
import json
import re
import requests
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from langchain.tools import tool
from google import genai
from google.genai import types
from requests.auth import HTTPBasicAuth

# Import settings globally
from app.config.settings import settings

logger = logging.getLogger(__name__)


# =============================================================================
# SIZE AND LIMIT CONSTANTS
# =============================================================================
MAX_FILE_SIZE_MB = 20  # Gemini file upload limit
LARGE_FILE_THRESHOLD_MB = 10  # Warn for files above this
MAX_VISIBLE_TEXT_CHARS = 100000  # ~100K chars for visible_text (prevent huge responses)
MAX_EXTRACTED_TEXT_FOR_POSTPROCESS = 50000  # Max text to run regex on (performance)


# =============================================================================
# INTELLIGENT DOCUMENT ANALYSIS PROMPT
# =============================================================================
DOCUMENT_ANALYSIS_PROMPT = """You are an intelligent document analyzer for a plumbing fixtures company (Flusso).
Analyze this document and provide structured information.

STEP 1: CLASSIFY THE DOCUMENT TYPE
Identify what type of document this is:
- "invoice": Sales invoices, purchase orders, receipts, order confirmations
- "spec_sheet": Product specification sheets, datasheets, technical documents
- "warranty_document": Warranty cards, warranty terms, coverage documents
- "installation_manual": Installation guides, setup instructions, user manuals
- "shipping_document": Shipping labels, packing slips, delivery receipts, BOL
- "return_authorization": RGA forms, return requests, RMA documents
- "agreement": Contracts, terms of service, sales agreements, NDAs
- "dealership_application": Dealer applications, partnership requests, reseller forms
- "license": Business licenses, certifications, permits, compliance documents
- "correspondence": Emails, letters, customer communications
- "catalog": Product catalogs, price lists, brochures
- "other": Any other document type

STEP 2: ANALYZE BASED ON DOCUMENT TYPE
Extract information relevant to the document type:

For "invoice" or "shipping_document":
- Vendor/seller information
- Order/PO/invoice number
- Date of purchase/shipment
- Customer/billing information
- Line items with products, quantities, prices
- Shipping details
- Total amounts

⚠️ IMPORTANT: Order numbers, PO numbers, and invoice numbers are NOT product model numbers.
Do NOT include order numbers in the "identifiers" section unless they clearly represent a product model.

For "spec_sheet" or "catalog":
- Product names and descriptions
- Model numbers and SKUs (NOT order numbers)
- Technical specifications (dimensions, materials, finishes)
- Features and capabilities
- Compatible parts/accessories

For "warranty_document":
- Warranty period/duration
- Coverage terms
- Exclusions
- Claim procedures
- Registration requirements

For "installation_manual":
- Product being installed
- Installation steps
- Required tools/parts
- Safety warnings
- Troubleshooting tips

For "return_authorization" or "correspondence":
- Reference numbers (RGA, ticket, case numbers)
- Customer information
- Issue description
- Resolution/action items

For "agreement" or "license":
- Parties involved
- Effective date
- Terms and conditions
- Key obligations
- Expiration date

For "dealership_application":
- Applicant information
- Business details
- Territory/region
- Application status

STEP 3: EXTRACT ALL VISIBLE TEXT
Extract all readable text from the document verbatim.

STEP 4: IDENTIFY PRODUCT-RELATED CODES (BONUS)
Look for and extract ANY product-related identifiers:
- Model numbers (e.g., 100.1170, HS6270MB, F2580CP, 260.2693T)
- Part numbers (e.g., 160.1168-9862, RP-12345)
- SKU codes
- UPC/EAN barcodes
- Serial numbers

⚠️ CRITICAL DISTINCTION - DO NOT CONFUSE THESE:
❌ DO NOT include in "model_numbers":
   - Order numbers (e.g., #69460-2, PO-12345)
   - Invoice numbers (e.g., INV-2024-001)
   - Tracking numbers (e.g., 1Z999AA10123456784)
   - These go in "order_numbers" or "tracking_numbers", NOT "model_numbers"

✅ Only include in "model_numbers":
   - Product model identifiers (e.g., 260.2693T, HS6270MB)
   - These typically appear on product labels, spec sheets, or product lists

OUTPUT FORMAT (MANDATORY JSON):
{
  "document_type": "<one of the types above>",
  "confidence": <0.0-1.0 how confident you are in the classification>,
  "description": "<2-3 sentence summary of what this document is about>",
  "extracted_data": {
    // Fields depend on document_type - include only relevant fields
    // For invoices: vendor, order_number, date, line_items, total, etc.
    // For spec sheets: product_name, model, specifications, etc.
    // For agreements: parties, terms, dates, etc.
  },
  "visible_text": "<all readable text from the document>",
  "identifiers": {
    "model_numbers": [],  // ✅ ONLY product model numbers, NOT order numbers
    "part_numbers": [],
    "serial_numbers": [],
    "order_numbers": [],  // ✅ Use this for purchase/order numbers
    "tracking_numbers": [],
    "sku_codes": [],
    "reference_numbers": []
  }
}

IMPORTANT:
- Be accurate about what you see, don't infer or guess
- Extract ALL text you can read
- Model/part numbers are BONUS info - extract what's clearly visible
- Focus on providing useful context for customer support"""


# Patterns for extracting product-related identifiers
PRODUCT_IDENTIFIER_PATTERNS = [
    r'\b(\d{3}\.\d{4}[A-Z]{0,3})\b',                 # 100.1050SB, 196.1280
    r'\b([A-Z]{2,4}\.\d{4}[A-Z]{0,3})\b',            # DKM.2420, CFB.2250
    r'\b(\d{2,3}\.[A-Z]{2,4}\.\d{4}[A-Z]{0,3})\b',   # 10.GGC.4026CP
    r'\b(\d{3}\.\d{2}[A-Z]{3,5})\b',                 # 160.16CSASG
    r'\b(\d{6}-\d{3}[A-Z]{0,4})\b',                  # 156297-435
    r'\b(\d{4}-\d{3}[A-Z]{0,3})\b',                  # 7764-441BB
    r'\b([A-Z]{2}\d{4}[A-Z]{2})\b',                  # HS6270MB
    r'\b([A-Z]{3}-\d{4}[A-Z]{0,2})\b',               # CFB-2250
    r'\b(RP-?\d{4,6})\b',                            # RP-12345 (part numbers)
    r'\b([A-Z]{1,3}\d{4,6}[A-Z]{0,3})\b',            # F2580CP, A1234
]


def _extract_identifiers_from_text(text: str) -> Dict[str, List[str]]:
    """
    Extract product identifiers from text using regex patterns.
    Post-processing step to catch any codes Gemini might have missed.
    """
    if not text:
        return {}
    
    found = []
    for pattern in PRODUCT_IDENTIFIER_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found.extend(matches)
    
    # Deduplicate and uppercase
    unique = list(set(m.upper() for m in found if len(m) >= 4))
    
    return {"additional_codes": unique[:20]} if unique else {}


def _parse_document_response(response_text: str) -> Dict[str, Any]:
    """
    Parse the document analysis response from Gemini.
    Handles both clean JSON and responses with markdown formatting.
    """
    if not response_text:
        return {
            "document_type": "unknown",
            "confidence": 0.0,
            "description": "Failed to analyze document",
            "extracted_data": {},
            "visible_text": "",
            "identifiers": {}
        }
    
    try:
        # Clean up response - remove markdown code blocks if present
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        parsed = json.loads(cleaned)
        
        # Ensure all expected fields exist
        return {
            "document_type": parsed.get("document_type", "unknown"),
            "confidence": parsed.get("confidence", 0.5),
            "description": parsed.get("description", ""),
            "extracted_data": parsed.get("extracted_data", {}),
            "visible_text": parsed.get("visible_text", ""),
            "identifiers": parsed.get("identifiers", {})
        }
        
    except json.JSONDecodeError as e:
        # Try to extract JSON from the response if it's embedded
        logger.warning(f"[DOC_ANALYZER] Initial JSON parse failed: {e}")
        
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx + 1]
                parsed = json.loads(json_str)
                logger.info(f"[DOC_ANALYZER] Successfully extracted embedded JSON")
                return {
                    "document_type": parsed.get("document_type", "unknown"),
                    "confidence": parsed.get("confidence", 0.5),
                    "description": parsed.get("description", ""),
                    "extracted_data": parsed.get("extracted_data", {}),
                    "visible_text": parsed.get("visible_text", ""),
                    "identifiers": parsed.get("identifiers", {})
                }
        except json.JSONDecodeError:
            pass
        
        # Final fallback
        logger.warning(f"[DOC_ANALYZER] Failed to parse JSON, using fallback")
        return {
            "document_type": "unknown",
            "confidence": 0.3,
            "description": response_text[:500],
            "extracted_data": {},
            "visible_text": response_text,
            "identifiers": {}
        }


def _download_attachment(url: str, name: str) -> str:
    """
    Download attachment to temp file and return path.
    
    IMPORTANT: Freshdesk uses two types of attachment URLs:
    1. Direct Freshdesk API URLs - require Basic Auth with API key
    2. S3 signed URLs (amazonaws.com) - already contain AWS signature, NO auth needed
    
    Adding auth to S3 signed URLs causes HTTP 400 errors!
    """
    try:
        logger.info(f"[DOC_ANALYZER] Downloading: {name}")
        
        # Detect if this is an S3 signed URL
        is_s3_signed_url = (
            "amazonaws.com" in url.lower() or 
            "X-Amz-Signature" in url or
            "x-amz-signature" in url.lower()
        )
        
        if is_s3_signed_url:
            # S3 signed URLs should NOT use authentication
            logger.info(f"[DOC_ANALYZER] S3 signed URL detected - no auth needed")
            response = requests.get(url, timeout=30)
        else:
            # Direct Freshdesk API URLs require Basic Auth
            auth = HTTPBasicAuth(settings.freshdesk_api_key, "X")
            response = requests.get(url, auth=auth, timeout=30)
        
        response.raise_for_status()
        
        # Create temp file with proper extension
        suffix = "." + name.split(".")[-1] if "." in name else ".pdf"
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        
        with open(path, "wb") as f:
            f.write(response.content)
        
        file_size_mb = len(response.content) / (1024 * 1024)
        logger.info(f"[DOC_ANALYZER] Downloaded to: {path} ({file_size_mb:.2f} MB)")
        return path, file_size_mb
        
    except Exception as e:
        logger.error(f"[DOC_ANALYZER] Download failed for {name}: {e}")
        raise


def _check_file_size(file_size_mb: float, filename: str) -> Tuple[bool, str]:
    """
    Check if file size is within acceptable limits.
    
    Returns:
        Tuple of (is_valid, message)
    """
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({file_size_mb:.1f}MB). Max allowed: {MAX_FILE_SIZE_MB}MB"
    elif file_size_mb > LARGE_FILE_THRESHOLD_MB:
        return True, f"Large file ({file_size_mb:.1f}MB) - analysis may take longer and text may be truncated"
    else:
        return True, "OK"


def _smart_truncate_text(text: str, max_chars: int = MAX_VISIBLE_TEXT_CHARS) -> Tuple[str, bool]:
    """
    Intelligently truncate text while preserving structure.
    
    Strategy:
    - If text is within limit, return as-is
    - If over limit, keep first 70% and last 30% with ellipsis in middle
    - This preserves headers/intro AND conclusions/signatures
    
    Returns:
        Tuple of (truncated_text, was_truncated)
    """
    if len(text) <= max_chars:
        return text, False
    
    # Calculate split points (70% beginning, 30% end)
    keep_start = int(max_chars * 0.70)
    keep_end = int(max_chars * 0.30)
    
    truncation_notice = f"\n\n[... DOCUMENT TRUNCATED - Showing first ~{keep_start//1000}K and last ~{keep_end//1000}K chars of {len(text)//1000}K total ...]\n\n"
    
    truncated = text[:keep_start] + truncation_notice + text[-keep_end:]
    
    logger.info(f"[DOC_ANALYZER] Text truncated: {len(text)} chars → {len(truncated)} chars")
    return truncated, True


def _get_page_estimate(file_size_mb: float) -> int:
    """
    Estimate number of pages based on file size.
    Average PDF: ~100KB per page (with images), ~30KB per page (text-only)
    We use 50KB as a middle estimate.
    """
    avg_page_size_mb = 0.05  # 50KB
    return max(1, int(file_size_mb / avg_page_size_mb))


@tool
def multimodal_document_analyzer_tool(
    attachments: List[Dict[str, Any]],
    focus: str = "general"
) -> Dict[str, Any]:
    """
    Intelligent Document Analyzer - Classifies and analyzes documents contextually.
    
    This tool performs:
    1. Document classification (invoice, spec sheet, warranty, agreement, etc.)
    2. Context-aware analysis based on document type
    3. Structured data extraction relevant to the document type
    4. Full text extraction
    5. Product identifier extraction (model/part/order numbers as bonus)
    
    Document Types Detected:
    - invoice: Sales invoices, purchase orders, receipts
    - spec_sheet: Product specifications, datasheets
    - warranty_document: Warranty terms and coverage
    - installation_manual: Setup guides, instructions
    - shipping_document: Labels, packing slips, BOL
    - return_authorization: RGA forms, RMA documents
    - agreement: Contracts, terms of service
    - dealership_application: Dealer/partner applications
    - license: Business licenses, certifications
    - correspondence: Emails, letters
    - catalog: Product catalogs, price lists
    - other: Any other document type
    
    Args:
        attachments: List of attachment dicts with 'attachment_url' and 'name'
        focus: Deprecated - kept for backwards compatibility
    
    Returns:
        Dict containing:
        - success: bool
        - documents: list of per-document analysis results
        - count: number of documents processed
        - summary: overall summary of document types found
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
        # Initialize Gemini client
        if not settings.gemini_api_key:
            return {"success": False, "message": "Missing GEMINI_API_KEY"}
             
        client = genai.Client(api_key=settings.gemini_api_key)
        
        documents = []
        temp_files = []
        all_identifiers = {
            "model_numbers": [],
            "part_numbers": [],
            "order_numbers": [],
            "serial_numbers": [],
            "tracking_numbers": [],
            "sku_codes": [],
            "reference_numbers": []
        }
        document_type_summary = {}
        
        for att in attachments:
            url = att.get("attachment_url")
            name = att.get("name", "unknown")
            
            if not url:
                logger.warning(f"[DOC_ANALYZER] No URL for attachment: {name}")
                continue
            
            try:
                # 1. Download to temp file (now returns file size too)
                local_path, file_size_mb = _download_attachment(url, name)
                temp_files.append(local_path)
                
                # 2. Check file size limits
                is_valid, size_message = _check_file_size(file_size_mb, name)
                if not is_valid:
                    logger.error(f"[DOC_ANALYZER] File rejected: {size_message}")
                    documents.append({
                        "filename": name,
                        "document_type": "error",
                        "confidence": 0.0,
                        "description": f"File rejected: {size_message}",
                        "extracted_data": {},
                        "visible_text": "",
                        "identifiers": {},
                        "status": "rejected",
                        "error": size_message,
                        "file_size_mb": file_size_mb
                    })
                    continue
                
                # Log warning for large files
                estimated_pages = _get_page_estimate(file_size_mb)
                if file_size_mb > LARGE_FILE_THRESHOLD_MB:
                    logger.warning(f"[DOC_ANALYZER] Large file: {name} ({file_size_mb:.1f}MB, ~{estimated_pages} pages)")
                
                # 3. Upload to Gemini Files API
                logger.info(f"[DOC_ANALYZER] Uploading {name} to Gemini ({file_size_mb:.2f}MB)")
                file_obj = client.files.upload(file=local_path)
                
                # 4. Call Gemini for intelligent analysis
                logger.info(f"[DOC_ANALYZER] Analyzing {name} with gemini-2.5-flash (~{estimated_pages} pages)")
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        types.Content(
                            parts=[
                                types.Part(text=DOCUMENT_ANALYSIS_PROMPT),
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
                        temperature=0.1
                    )
                )
                
                # 5. Parse response
                response_text = response.text if response.text else ""
                analysis = _parse_document_response(response_text)
                
                # 6. Smart truncation of visible_text for large documents
                visible_text = analysis.get("visible_text", "")
                visible_text, was_truncated = _smart_truncate_text(visible_text)
                
                # 7. Post-process: Extract additional identifiers from visible text
                # Use truncated text for regex to avoid performance issues
                text_for_regex = visible_text[:MAX_EXTRACTED_TEXT_FOR_POSTPROCESS]
                additional = _extract_identifiers_from_text(text_for_regex)
                
                # Merge identifiers
                identifiers = analysis.get("identifiers", {})
                if additional.get("additional_codes"):
                    existing_codes = set()
                    for key in identifiers:
                        if isinstance(identifiers[key], list):
                            existing_codes.update(identifiers[key])
                    
                    new_codes = [c for c in additional["additional_codes"] if c not in existing_codes]
                    if new_codes:
                        identifiers["additional_codes"] = new_codes
                
                # Collect all identifiers
                for key in all_identifiers:
                    if key in identifiers and isinstance(identifiers[key], list):
                        all_identifiers[key].extend(identifiers[key])
                
                # Track document types
                doc_type = analysis.get("document_type", "unknown")
                document_type_summary[doc_type] = document_type_summary.get(doc_type, 0) + 1
                
                # Build result with metadata about size/truncation
                result_doc = {
                    "filename": name,
                    "document_type": doc_type,
                    "confidence": analysis.get("confidence", 0.5),
                    "description": analysis.get("description", ""),
                    "extracted_data": analysis.get("extracted_data", {}),
                    "visible_text": visible_text,
                    "identifiers": identifiers,
                    "status": "success",
                    "file_size_mb": round(file_size_mb, 2),
                    "estimated_pages": estimated_pages
                }
                
                # Add truncation warning if applicable
                if was_truncated:
                    result_doc["text_truncated"] = True
                    result_doc["truncation_note"] = f"Document text was truncated (>{MAX_VISIBLE_TEXT_CHARS//1000}K chars)"
                
                documents.append(result_doc)
                
                truncation_flag = " [TRUNCATED]" if was_truncated else ""
                logger.info(f"[DOC_ANALYZER] ✓ {name}: type={doc_type}, confidence={analysis.get('confidence', 0):.0%}, {file_size_mb:.2f}MB{truncation_flag}")
                
            except Exception as e:
                logger.error(f"[DOC_ANALYZER] Failed to process {name}: {e}", exc_info=True)
                documents.append({
                    "filename": name,
                    "document_type": "error",
                    "confidence": 0.0,
                    "description": f"Processing failed: {str(e)}",
                    "extracted_data": {},
                    "visible_text": "",
                    "identifiers": {},
                    "status": "error",
                    "error": str(e)
                })
        
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"[DOC_ANALYZER] Failed to delete temp file {temp_file}: {e}")
        
        # Deduplicate all_identifiers
        for key in all_identifiers:
            all_identifiers[key] = list(set(all_identifiers[key]))
        
        # Build summary
        type_summary = ", ".join([f"{count} {dtype}" for dtype, count in document_type_summary.items()])
        
        logger.info(f"[DOC_ANALYZER] ✅ Completed: {len(documents)} document(s) - {type_summary}")
        
        return {
            "success": True,
            "documents": documents,
            "count": len(documents),
            "document_types": document_type_summary,
            "all_identifiers": all_identifiers,
            "message": f"Analyzed {len(documents)} document(s): {type_summary}"
        }
        
    except Exception as e:
        logger.error(f"[DOC_ANALYZER] Critical error: {e}", exc_info=True)
        return {
            "success": False,
            "documents": [],
            "count": 0,
            "message": f"Document analysis failed: {str(e)}"
        }