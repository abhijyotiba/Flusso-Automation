from langchain.tools import tool
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

ATTACHMENT_TYPES = {
    "invoice": ["invoice", "bill"],
    "purchase_order": ["po_", "po-", "purchase order", "sales order"],
    "receipt": ["receipt", "sales slip"],
    "form": ["application", "form"],
    "product_box": ["box", "label", "packaging"],
    "damage": ["damage", "broken", "crack", "dent"],
}

def simple_classify(name: str, content_type: str) -> str:
    lname = name.lower()

    if content_type.startswith("image/"):
        for key in ["damage", "product_box"]:
            if any(k in lname for k in ATTACHMENT_TYPES[key]):
                return key
        return "image"

    if content_type.startswith("application/pdf") or "word" in content_type or "excel" in content_type:
        for key in ["purchase_order", "invoice", "receipt", "form"]:
            if any(k in lname for k in ATTACHMENT_TYPES[key]):
                return key
        return "document"

    return "unknown"


@tool
def attachment_type_classifier_tool(attachments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Lightweight classifier to help the agent decide:
        - Which attachments are images (→ OCR)
        - Which attachments are documents (→ Gemini Pro)
    """

    results = []
    for att in attachments:
        name = att.get("name") or att.get("filename")
        ct = att.get("content_type", "")
        detected = simple_classify(name, ct)

        results.append({
            "name": name,
            "content_type": ct,
            "detected_type": detected
        })

    return {
        "success": True,
        "attachments": results
    }
