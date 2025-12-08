"""
Tools Package for ReACT Agent
Exports all available tools
"""

from app.tools.product_search import product_search_tool
from app.tools.document_search import document_search_tool
from app.tools.vision_search import vision_search_tool
from app.tools.past_tickets import past_tickets_search_tool
from app.tools.attachment_analyzer import attachment_analyzer_tool
from app.tools.finish import finish_tool

__all__ = [
    "product_search_tool",
    "document_search_tool",
    "vision_search_tool",
    "past_tickets_search_tool",
    "attachment_analyzer_tool",
    "finish_tool"
]

# Tool registry for easy access
AVAILABLE_TOOLS = {
    "product_search_tool": product_search_tool,
    "document_search_tool": document_search_tool,
    "vision_search_tool": vision_search_tool,
    "past_tickets_search_tool": past_tickets_search_tool,
    "attachment_analyzer_tool": attachment_analyzer_tool,
    "finish_tool": finish_tool
}