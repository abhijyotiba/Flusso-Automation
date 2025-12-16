"""
Intelligent Image Analyzer Tool
Uses Gemini Vision to classify and analyze images contextually.

This tool is NOT just OCR - it's a smart analyzer that:
1. Classifies the image type (receipt, product photo, damaged part, package, etc.)
2. Analyzes contextually based on what kind of image it is
3. Extracts relevant information for that image type
4. Model numbers are a bonus, not the primary goal
"""

from langchain.tools import tool
from google import genai
from google.genai import types
import httpx
import logging
import json
import re
from typing import List, Dict, Any, Optional

from app.config.settings import settings

# Configure logger
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE CLASSIFICATION AND ANALYSIS PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

IMAGE_ANALYSIS_PROMPT = """You are an intelligent image analyzer for a plumbing fixtures support system.

STEP 1: CLASSIFY THE IMAGE
First, determine what type of image this is:
- "purchase_receipt" - Invoice, receipt, order confirmation, packing slip
- "product_photo" - Photo of a complete product (faucet, shower head, valve, etc.)
- "product_part" - Photo of a specific part or component
- "damaged_item" - Photo showing damage, defect, or wear on a product/part
- "packaging" - Photo of product packaging, box, shipping label
- "installation" - Photo of installed product or installation process
- "serial_label" - Close-up of serial number, model label, or barcode
- "other" - Anything else (documents, general photos, etc.)

STEP 2: ANALYZE BASED ON TYPE
Based on the classification, extract the most relevant information:

For "purchase_receipt":
- Store/vendor name
- Order number, PO number, invoice number
- Date of purchase
- Line items with quantities and prices
- Total amount
- Customer information if visible

For "product_photo" or "product_part":
- What product/part is shown (e.g., "diverter valve", "shower head", "faucet handle")
- Visible features, finish (chrome, brushed nickel, etc.)
- Approximate condition
- Any visible model numbers or labels

For "damaged_item":
- What is damaged (describe the item)
- Type of damage (crack, leak, corrosion, broken piece, wear)
- Severity (minor, moderate, severe)
- Affected area/component

For "packaging":
- Product name if visible
- Model/SKU numbers
- Shipping tracking numbers
- Sender/recipient info

For "serial_label":
- Model number
- Serial number
- Part numbers
- Manufacturer info
- Date codes

For "installation":
- What is being installed
- Installation stage (in progress, completed)
- Any visible issues

STEP 3: EXTRACT TEXT (if present)
If there is any visible text in the image, extract it verbatim.

⚠️ CRITICAL DISTINCTION - DO NOT CONFUSE THESE:
When extracting identifiers, be careful to distinguish:

❌ DO NOT include in "model_numbers":
   - Order numbers from shipping labels (e.g., #69460-2, PO-12345)
   - Invoice numbers (e.g., INV-2024-001)
   - Tracking numbers (e.g., 1Z999AA10123456784)
   - These go in "order_numbers" or "tracking_numbers", NOT "model_numbers"

✅ Only include in "model_numbers":
   - Product model identifiers from product labels (e.g., 260.2693T, HS6270MB)
   - Part numbers on the product itself
   - SKU codes on product packaging

OUTPUT FORMAT (MANDATORY JSON):
{
  "image_type": "<one of the types above>",
  "confidence": <0.0-1.0 how confident you are in the classification>,
  "description": "<2-3 sentence description of what the image shows>",
  "extracted_data": {
    // Fields depend on image_type - include only relevant fields
    // For receipts: vendor, order_number, date, items, total, etc.
    // For products: product_type, features, finish, condition, etc.
    // For damage: item, damage_type, severity, affected_area, etc.
  },
  "visible_text": "<all text visible in image, verbatim>",
  "identifiers": {
    "model_numbers": [],  // ✅ ONLY product models, NOT order numbers
    "serial_numbers": [],
    "order_numbers": [],  // ✅ Use this for shipping/order numbers
    "tracking_numbers": []
  }
}

IMPORTANT:
- Be accurate about what you see, don't infer or guess
- If you can't clearly see something, say so
- Model numbers are BONUS info - don't force extraction if not clearly visible
- Focus on providing useful context for customer support
- DO NOT confuse order numbers with product model numbers"""


# Flusso-specific model number patterns for post-processing
FLUSSO_MODEL_PATTERNS = [
    r'\b(\d{3}\.\d{4}[A-Z]{0,3})\b',                 # 100.1050SB, 196.1280
    r'\b([A-Z]{2,4}\.\d{4}[A-Z]{0,3})\b',            # DKM.2420, CFB.2250
    r'\b(\d{2,3}\.[A-Z]{2,4}\.\d{4}[A-Z]{0,3})\b',   # 10.GGC.4026CP
    r'\b(\d{3}\.\d{2}[A-Z]{3,5})\b',                 # 160.16CSASG
    r'\b(\d{6}-\d{3}[A-Z]{0,2})\b',                  # 156297-435
    r'\b(\d{4}-\d{3}[A-Z]{0,3})\b',                  # 7764-441BB
    r'\b([A-Z]{2}\d{4}[A-Z]{2})\b',                  # HS6270MB
    r'\b([A-Z]{3}-\d{4}[A-Z]{0,2})\b',               # CFB-2250
]


def _extract_flusso_model_numbers(text: str) -> List[str]:
    """
    Extract valid Flusso model numbers from text using regex patterns.
    This is a post-processing step to catch any models Gemini might have missed.
    """
    if not text:
        return []
    
    found_models = []
    for pattern in FLUSSO_MODEL_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found_models.extend(matches)
    
    # Deduplicate and uppercase
    unique_models = list(set(m.upper() for m in found_models))
    return unique_models[:10]


def _parse_analysis_response(response_text: str) -> Dict[str, Any]:
    """
    Parse the image analysis response from Gemini.
    Handles both clean JSON and responses with markdown formatting.
    """
    if not response_text:
        return {
            "image_type": "unknown",
            "confidence": 0.0,
            "description": "Failed to analyze image",
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
            "image_type": parsed.get("image_type", "unknown"),
            "confidence": parsed.get("confidence", 0.5),
            "description": parsed.get("description", ""),
            "extracted_data": parsed.get("extracted_data", {}),
            "visible_text": parsed.get("visible_text", ""),
            "identifiers": parsed.get("identifiers", {})
        }
        
    except json.JSONDecodeError as e:
        # Try to extract JSON from the response if it's embedded
        logger.warning(f"[IMAGE_ANALYZER] Initial JSON parse failed: {e}")
        
        # Try to find JSON object in the response
        try:
            # Look for JSON starting with { and ending with }
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx + 1]
                parsed = json.loads(json_str)
                logger.info(f"[IMAGE_ANALYZER] Successfully extracted embedded JSON")
                return {
                    "image_type": parsed.get("image_type", "unknown"),
                    "confidence": parsed.get("confidence", 0.5),
                    "description": parsed.get("description", ""),
                    "extracted_data": parsed.get("extracted_data", {}),
                    "visible_text": parsed.get("visible_text", ""),
                    "identifiers": parsed.get("identifiers", {})
                }
        except json.JSONDecodeError:
            pass
        
        # Final fallback: return the raw text as description
        logger.warning(f"[IMAGE_ANALYZER] Failed to parse JSON, using fallback")
        return {
            "image_type": "unknown",
            "confidence": 0.3,
            "description": response_text[:500],
            "extracted_data": {},
            "visible_text": response_text,
            "identifiers": {}
        }


@tool
def ocr_image_analyzer_tool(image_urls: List[str]) -> Dict[str, Any]:
    """
    Intelligent Image Analyzer - Classifies and analyzes images contextually.
    
    This is NOT just OCR. It performs:
    1. Image classification (receipt, product photo, damaged part, package, etc.)
    2. Context-aware analysis based on image type
    3. Structured data extraction relevant to the image type
    4. Text extraction (if text is present)
    5. Model/serial number identification (bonus, not primary goal)
    
    Image Types Detected:
    - purchase_receipt: Invoices, receipts, order confirmations
    - product_photo: Photos of complete products
    - product_part: Photos of specific parts/components
    - damaged_item: Photos showing damage or defects
    - packaging: Product boxes, shipping labels
    - installation: Installed products or installation process
    - serial_label: Close-ups of labels, barcodes, model plates
    - other: Anything else
    
    Args:
        image_urls: List of public HTTP/HTTPS URLs to images
        
    Returns:
        Dict containing:
        - success: bool
        - count: number of images processed
        - results: list of per-image analysis results
        - summary: overall summary of what was found
    """
    
    # 1. Initialize Client
    if not settings.gemini_api_key:
        return {"success": False, "error": "Missing GEMINI_API_KEY in settings"}

    try:
        client = genai.Client(api_key=settings.gemini_api_key)
    except Exception as e:
        logger.error(f"[IMAGE_ANALYZER] Failed to initialize Gemini client: {e}")
        return {"success": False, "error": f"Client init failed: {str(e)}"}

    results = []
    all_model_numbers = []
    all_order_numbers = []
    image_type_summary = {}

    # 2. Process each image
    for index, url in enumerate(image_urls):
        try:
            logger.info(f"[IMAGE_ANALYZER] Processing image {index + 1}/{len(image_urls)}: {url[:80]}...")
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            # Download image
            with httpx.Client(timeout=30.0, follow_redirects=True) as http_client:
                image_resp = http_client.get(url, headers=headers)
                image_resp.raise_for_status()
                
                mime_type = image_resp.headers.get("content-type", "image/jpeg")
                image_bytes = image_resp.content

            # 3. Send to Gemini for intelligent analysis
            # Using gemini-2.5-flash for better vision capabilities
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Content(
                        parts=[
                            types.Part(text=IMAGE_ANALYSIS_PROMPT),
                            types.Part.from_bytes(
                                data=image_bytes,
                                mime_type=mime_type
                            )
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1  # Low temp for accurate analysis
                )
            )
            
            # 4. Parse response
            response_text = response.text if response.text else ""
            analysis = _parse_analysis_response(response_text)
            
            # 5. Post-process: Apply regex to catch any missed model numbers
            visible_text = analysis.get("visible_text", "")
            regex_models = _extract_flusso_model_numbers(visible_text)
            
            # Merge regex-found models with Gemini-detected ones
            identifiers = analysis.get("identifiers", {})
            gemini_models = identifiers.get("model_numbers", [])
            all_found_models = list(set(gemini_models + regex_models))
            
            if all_found_models:
                identifiers["model_numbers"] = all_found_models
                analysis["identifiers"] = identifiers
            
            # Collect all identifiers across images
            all_model_numbers.extend(identifiers.get("model_numbers", []))
            all_order_numbers.extend(identifiers.get("order_numbers", []))
            
            # Track image types
            img_type = analysis.get("image_type", "unknown")
            image_type_summary[img_type] = image_type_summary.get(img_type, 0) + 1
            
            # Add to results
            results.append({
                "image_index": index + 1,
                "image_url": url,
                "image_type": img_type,
                "confidence": analysis.get("confidence", 0.5),
                "description": analysis.get("description", ""),
                "extracted_data": analysis.get("extracted_data", {}),
                "visible_text": visible_text,
                "identifiers": identifiers,
                "status": "success"
            })
            
            logger.info(f"[IMAGE_ANALYZER] ✓ Image {index + 1}: type={img_type}, confidence={analysis.get('confidence', 0):.0%}")

        except httpx.HTTPStatusError as e:
            logger.error(f"[IMAGE_ANALYZER] Failed to download image {url}: {e}")
            results.append({
                "image_index": index + 1,
                "image_url": url,
                "image_type": "error",
                "description": f"Download failed (Status {e.response.status_code})",
                "status": "error",
                "error": str(e)
            })
            
        except Exception as e:
            logger.error(f"[IMAGE_ANALYZER] Analysis failed for {url}: {e}")
            results.append({
                "image_index": index + 1,
                "image_url": url,
                "image_type": "error",
                "description": f"Analysis failed: {str(e)}",
                "status": "error",
                "error": str(e)
            })

    # 6. Build summary
    successful = [r for r in results if r.get("status") == "success"]
    
    summary_parts = []
    if image_type_summary:
        type_str = ", ".join([f"{count} {img_type}" for img_type, count in image_type_summary.items()])
        summary_parts.append(f"Analyzed {len(successful)} image(s): {type_str}")
    
    if all_model_numbers:
        unique_models = list(set(all_model_numbers))
        summary_parts.append(f"Model numbers found: {', '.join(unique_models[:5])}")
    
    if all_order_numbers:
        unique_orders = list(set(all_order_numbers))
        summary_parts.append(f"Order numbers found: {', '.join(unique_orders[:5])}")

    # 7. Construct Final Output
    return {
        "success": True,
        "count": len(results),
        "successful_count": len(successful),
        "results": results,
        "summary": " | ".join(summary_parts) if summary_parts else "No significant data extracted",
        "all_identifiers": {
            "model_numbers": list(set(all_model_numbers)),
            "order_numbers": list(set(all_order_numbers))
        },
        "image_types": image_type_summary
    }