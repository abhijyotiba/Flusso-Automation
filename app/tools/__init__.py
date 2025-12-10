"""
Tools Package for ReACT Agent
"""

# IMPORT FROM THE CSV WRAPPER
from app.tools.product_search_from_csv import product_search_tool

# Import other tools
from app.tools.document_search import document_search_tool
from app.tools.vision_search import vision_search_tool
from app.tools.past_tickets import past_tickets_search_tool
from app.tools.attachment_analyzer import attachment_analyzer_tool
from app.tools.finish import finish_tool
from app.tools.multimodal_document_analyzer import multimodal_document_analyzer_tool
from app.tools.ocr_image_analyzer import ocr_image_analyzer_tool
from app.tools.attachment_classifier_tool import attachment_type_classifier_tool

__all__ = [
    "product_search_tool",
    "document_search_tool",
    "vision_search_tool",
    "past_tickets_search_tool",
    "attachment_analyzer_tool",
    "finish_tool",
    "multimodal_document_analyzer_tool",
    "ocr_image_analyzer_tool",
    "attachment_type_classifier_tool"
]

AVAILABLE_TOOLS = {
    "product_search_tool": product_search_tool,
    "document_search_tool": document_search_tool,
    "vision_search_tool": vision_search_tool,
    "past_tickets_search_tool": past_tickets_search_tool,
    "attachment_analyzer_tool": attachment_analyzer_tool,
    "finish_tool": finish_tool,
    "multimodal_document_analyzer_tool": multimodal_document_analyzer_tool,
    "ocr_image_analyzer_tool": ocr_image_analyzer_tool,
    "attachment_type_classifier_tool": attachment_type_classifier_tool
}