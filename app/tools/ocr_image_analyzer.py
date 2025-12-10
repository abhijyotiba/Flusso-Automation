from langchain.tools import tool
from google import genai
from google.genai import types
import httpx
import logging
from typing import List, Dict, Any

from app.config.settings import settings

# Configure logger
logger = logging.getLogger(__name__)

@tool
def ocr_image_analyzer_tool(image_urls: List[str]) -> Dict[str, Any]:
    """
    Performs OCR (Optical Character Recognition) on images using Gemini 1.5/2.0 Flash.
    
    This tool downloads images from the provided URLs, sends them to the Gemini API,
    and returns the extracted text.
    
    Args:
        image_urls (List[str]): A list of public HTTP/HTTPS URLs to images.
        
    Returns:
        Dict: Contains 'success', 'results', and crucially 'combined_text' 
              which is a single string of all extracted text for easy reading.
    """
    
    # 1. Initialize Client
    if not settings.gemini_api_key:
        return {"success": False, "error": "Missing GEMINI_API_KEY in settings"}

    try:
        client = genai.Client(api_key=settings.gemini_api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        return {"success": False, "error": f"Client init failed: {str(e)}"}

    results = []
    combined_text_parts = []

    # 2. Process each image
    for index, url in enumerate(image_urls):
        try:
            logger.info(f"Downloading image for OCR: {url}")
            
            # Use specific headers to avoid 403s from picky servers/S3
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            # Increased timeout and added follow_redirects for robustness
            with httpx.Client(timeout=20.0, follow_redirects=True) as http_client:
                image_resp = http_client.get(url, headers=headers)
                image_resp.raise_for_status()
                
                mime_type = image_resp.headers.get("content-type", "image/jpeg")
                image_bytes = image_resp.content

            # 3. Send to Gemini
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    types.Content(
                        parts=[
                            types.Part(text="Extract ALL text from this image. Return the text exactly as seen."),
                            types.Part.from_bytes(
                                data=image_bytes,
                                mime_type=mime_type
                            )
                        ]
                    )
                ]
            )
            
            # Extract text safely
            extracted_text = response.text if response.text else "[No text found in image]"
            
            # Add to structured results
            results.append({
                "image_url": url,
                "text": extracted_text,
                "status": "success"
            })
            
            # Add to formatted string for the Agent
            combined_text_parts.append(f"--- IMAGE {index + 1} TEXT ---\n{extracted_text}\n")

        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to download image {url}: {e}")
            error_msg = f"Download failed (Status {e.response.status_code})"
            results.append({"image_url": url, "text": "", "status": "error", "error": error_msg})
            combined_text_parts.append(f"--- IMAGE {index + 1} ERROR ---\n{error_msg}\n")
            
        except Exception as e:
            logger.error(f"OCR failed for {url}: {e}")
            results.append({"image_url": url, "text": "", "status": "error", "error": str(e)})
            combined_text_parts.append(f"--- IMAGE {index + 1} ERROR ---\n{str(e)}\n")

    # 4. Construct Final Output
    # The 'combined_text' field is critical for the ReACT agent to "see" the output immediately.
    final_output = {
        "success": True, 
        "count": len(results),
        "results": results,
        "combined_text": "\n".join(combined_text_parts)
    }
    
    return final_output