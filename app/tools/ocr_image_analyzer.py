from langchain.tools import tool
from google import genai
import os

@tool
def ocr_image_analyzer_tool(image_urls: list):
    """
    OCR using Gemini 1.5 Flash Vision.
    Useful for:
     - Product box labels
     - Receipts photos
     - Damaged part images
    """

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    result = []

    for url in image_urls:
        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                {"parts": [{
                    "text":
                    "Extract ALL text from this image. Return lines only, no commentary."
                },{
                    "image_url": {"url": url}
                }]}
            ]
        )

        result.append({
            "image_url": url,
            "text": resp.text
        })

    return {"success": True, "results": result}
