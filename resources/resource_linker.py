"""
Resource Linker Service
Extracts and formats direct links to product resources (documents, videos, images)
for inclusion in customer responses and agent notes.

Similar to Agent Console's data_loader.py but optimized for response generation.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Base URL for Flusso resources
FLUSSO_BASE_URL = "https://flussofaucets.com"


@dataclass
class ProductResources:
    """Container for all product resources with direct URLs."""
    
    model_no: str
    product_title: str = ""
    
    # Documents (PDFs)
    spec_sheet: Optional[Dict[str, str]] = None
    installation_manual: Optional[Dict[str, str]] = None
    parts_diagram: Optional[Dict[str, str]] = None
    
    # Videos
    installation_video: Optional[str] = None
    operational_video: Optional[str] = None
    lifestyle_video: Optional[str] = None
    
    # Images
    product_image: Optional[str] = None
    
    # Product page
    product_page_url: Optional[str] = None
    collection_url: Optional[str] = None
    
    def has_documents(self) -> bool:
        """Check if any documents are available."""
        return bool(self.spec_sheet or self.installation_manual or self.parts_diagram)
    
    def has_videos(self) -> bool:
        """Check if any videos are available."""
        return bool(self.installation_video or self.operational_video or self.lifestyle_video)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_no": self.model_no,
            "product_title": self.product_title,
            "documents": {
                "spec_sheet": self.spec_sheet,
                "installation_manual": self.installation_manual,
                "parts_diagram": self.parts_diagram,
            },
            "videos": {
                "installation": self.installation_video,
                "operational": self.operational_video,
                "lifestyle": self.lifestyle_video,
            },
            "images": {
                "product_image": self.product_image,
            },
            "links": {
                "product_page": self.product_page_url,
                "collection": self.collection_url,
            }
        }


def _ensure_https(url: str) -> str:
    """Ensure URL has https:// prefix."""
    if not url:
        return ""
    url = str(url).strip()
    if url and not url.startswith("http"):
        return f"https://{url}"
    return url


def extract_product_resources(product: Dict[str, Any]) -> ProductResources:
    """
    Extract all resource links from a product dictionary.
    
    Works with product data from:
    - product_catalog.py (main workflow)
    - product_catalog_tool results
    - identified_product from state
    
    Args:
        product: Product dictionary with URLs and file names
        
    Returns:
        ProductResources object with all available links
    """
    model_no = product.get("model_no", "") or product.get("Model_NO", "") or ""
    title = product.get("title", "") or product.get("Product_Title", "") or product.get("product_title", "") or ""
    
    resources = ProductResources(
        model_no=model_no,
        product_title=title
    )
    
    # === DOCUMENTS ===
    # Spec Sheet
    spec_url = product.get("spec_sheet_url") or product.get("Spec_Sheet_Full_URL") or ""
    spec_file = product.get("spec_sheet_file") or product.get("Spec_Sheet_File_Name") or ""
    if spec_url:
        resources.spec_sheet = {
            "url": _ensure_https(spec_url),
            "filename": spec_file,
            "title": "Specification Sheet"
        }
    
    # Installation Manual
    install_url = product.get("install_manual_url") or product.get("Installation_manual_Full_URL") or ""
    install_file = product.get("install_manual_file") or product.get("Installation_Manual_File_Name") or ""
    if install_url:
        resources.installation_manual = {
            "url": _ensure_https(install_url),
            "filename": install_file,
            "title": "Installation Manual"
        }
    
    # Parts Diagram
    parts_url = product.get("parts_diagram_url") or product.get("Part_Diagram_Full_URL") or ""
    parts_file = product.get("parts_diagram_file") or product.get("Parts_Diagram_File_Name") or ""
    if parts_url:
        resources.parts_diagram = {
            "url": _ensure_https(parts_url),
            "filename": parts_file,
            "title": "Parts Diagram"
        }
    
    # === VIDEOS ===
    resources.installation_video = _ensure_https(
        product.get("install_video_url") or product.get("Installation_video_Link") or ""
    ) or None
    
    resources.operational_video = _ensure_https(
        product.get("operational_video_url") or product.get("Operational_Video_Link") or ""
    ) or None
    
    resources.lifestyle_video = _ensure_https(
        product.get("lifestyle_video_url") or product.get("Lifestyle_Video_Link") or ""
    ) or None
    
    # === IMAGES ===
    resources.product_image = _ensure_https(
        product.get("image_url") or product.get("Image_URL") or ""
    ) or None
    
    # === PRODUCT PAGE ===
    resources.product_page_url = _ensure_https(
        product.get("product_url") or product.get("product_url") or ""
    ) or None
    
    # === COLLECTION ===
    collection_url = product.get("collection_url") or product.get("Collection_URL") or ""
    if collection_url:
        if collection_url.startswith("/"):
            resources.collection_url = f"{FLUSSO_BASE_URL}{collection_url}"
        else:
            resources.collection_url = _ensure_https(collection_url)
    
    return resources


def format_resources_html(resources: ProductResources, compact: bool = False) -> str:
    """
    Format product resources as HTML for inclusion in responses.
    
    Args:
        resources: ProductResources object
        compact: If True, use minimal styling
        
    Returns:
        HTML string with clickable resource links
    """
    if not resources.has_documents() and not resources.has_videos():
        return ""
    
    sections = []
    
    # === DOCUMENTS SECTION ===
    doc_links = []
    if resources.spec_sheet:
        doc_links.append(f"""
            <a href="{resources.spec_sheet['url']}" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; padding: 8px 12px; background: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 6px; text-decoration: none; color: #0369a1; font-size: 13px; margin: 4px;">
                📄 Spec Sheet
            </a>""")
    
    if resources.installation_manual:
        doc_links.append(f"""
            <a href="{resources.installation_manual['url']}" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; padding: 8px 12px; background: #f0fdf4; border: 1px solid #22c55e; border-radius: 6px; text-decoration: none; color: #166534; font-size: 13px; margin: 4px;">
                📘 Installation Manual
            </a>""")
    
    if resources.parts_diagram:
        doc_links.append(f"""
            <a href="{resources.parts_diagram['url']}" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; padding: 8px 12px; background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px; text-decoration: none; color: #92400e; font-size: 13px; margin: 4px;">
                🔧 Parts Diagram
            </a>""")
    
    if doc_links:
        sections.append(f"""
        <div style="margin-bottom: 12px;">
            <div style="font-weight: 600; color: #374151; margin-bottom: 8px; font-size: 13px;">📁 Documents:</div>
            <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                {''.join(doc_links)}
            </div>
        </div>""")
    
    # === VIDEOS SECTION ===
    video_links = []
    if resources.installation_video:
        video_links.append(f"""
            <a href="{resources.installation_video}" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; padding: 8px 12px; background: #fef2f2; border: 1px solid #ef4444; border-radius: 6px; text-decoration: none; color: #dc2626; font-size: 13px; margin: 4px;">
                🎬 Installation Video
            </a>""")
    
    if resources.operational_video:
        video_links.append(f"""
            <a href="{resources.operational_video}" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; padding: 8px 12px; background: #f5f3ff; border: 1px solid #8b5cf6; border-radius: 6px; text-decoration: none; color: #7c3aed; font-size: 13px; margin: 4px;">
                ▶️ Product Demo
            </a>""")
    
    if resources.lifestyle_video:
        video_links.append(f"""
            <a href="{resources.lifestyle_video}" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; padding: 8px 12px; background: #fdf4ff; border: 1px solid #d946ef; border-radius: 6px; text-decoration: none; color: #a21caf; font-size: 13px; margin: 4px;">
                🎥 Lifestyle Video
            </a>""")
    
    if video_links:
        sections.append(f"""
        <div style="margin-bottom: 12px;">
            <div style="font-weight: 600; color: #374151; margin-bottom: 8px; font-size: 13px;">🎬 Videos:</div>
            <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                {''.join(video_links)}
            </div>
        </div>""")
    
    # === PRODUCT PAGE ===
    if resources.product_page_url:
        sections.append(f"""
        <div>
            <a href="{resources.product_page_url}" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; padding: 8px 12px; background: #1e3a5f; border-radius: 6px; text-decoration: none; color: white; font-size: 13px;">
                🌐 View Product Page
            </a>
        </div>""")
    
    if not sections:
        return ""
    
    # Wrap in container
    html = f"""
    <div style="margin-top: 16px; padding: 16px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;">
        <div style="font-weight: bold; color: #1e3a5f; margin-bottom: 12px; font-size: 14px;">
            📎 Quick Reference Links for {resources.model_no}
        </div>
        {''.join(sections)}
    </div>"""
    
    return html


def format_resources_text(resources: ProductResources) -> str:
    """
    Format product resources as plain text with URLs.
    Useful for including in LLM context or non-HTML responses.
    
    Args:
        resources: ProductResources object
        
    Returns:
        Plain text string with resource links
    """
    if not resources.has_documents() and not resources.has_videos():
        return ""
    
    lines = [f"\n📎 QUICK REFERENCE LINKS ({resources.model_no}):"]
    
    # Documents
    if resources.spec_sheet:
        lines.append(f"  📄 Spec Sheet: {resources.spec_sheet['url']}")
    if resources.installation_manual:
        lines.append(f"  📘 Installation Manual: {resources.installation_manual['url']}")
    if resources.parts_diagram:
        lines.append(f"  🔧 Parts Diagram: {resources.parts_diagram['url']}")
    
    # Videos
    if resources.installation_video:
        lines.append(f"  🎬 Installation Video: {resources.installation_video}")
    if resources.operational_video:
        lines.append(f"  ▶️ Product Demo: {resources.operational_video}")
    if resources.lifestyle_video:
        lines.append(f"  🎥 Lifestyle Video: {resources.lifestyle_video}")
    
    # Product page
    if resources.product_page_url:
        lines.append(f"  🌐 Product Page: {resources.product_page_url}")
    
    return "\n".join(lines)


def get_resources_from_state(state: Dict[str, Any]) -> Optional[ProductResources]:
    """
    Extract product resources from workflow state.
    
    Looks for product info in:
    1. identified_product (from ReACT agent)
    2. source_products (first high-confidence match)
    3. gathered_documents metadata
    
    Args:
        state: Workflow state dictionary
        
    Returns:
        ProductResources if found, None otherwise
    """
    # Try identified_product first (set by ReACT agent)
    identified = state.get("identified_product")
    if identified and isinstance(identified, dict):
        logger.info(f"[RESOURCE_LINKER] Found identified_product: {identified.get('model_no') or identified.get('model')}")
        logger.debug(f"[RESOURCE_LINKER] identified_product keys: {list(identified.keys())}")
        
        # Check if it has any URL fields (non-empty strings)
        url_fields = ["spec_sheet_url", "install_manual_url", "parts_diagram_url",
                      "Spec_Sheet_Full_URL", "Installation_manual_Full_URL", "Part_Diagram_Full_URL",
                      "install_video_url", "product_url", "image_url"]
        has_urls = any(identified.get(k) and str(identified.get(k)).strip() for k in url_fields)
        
        if has_urls:
            logger.info(f"[RESOURCE_LINKER] Product has resource URLs, extracting...")
            return extract_product_resources(identified)
        else:
            # Still try to extract - maybe it has model_no and we can use that
            model = identified.get("model_no") or identified.get("model")
            if model:
                logger.info(f"[RESOURCE_LINKER] Product {model} has no URLs in identified_product, trying to lookup from catalog...")
                # Try to get full product data from catalog
                try:
                    from app.services.product_catalog import ensure_catalog_loaded
                    catalog = ensure_catalog_loaded()
                    full_product = catalog.search_exact_model(model)
                    if full_product:
                        logger.info(f"[RESOURCE_LINKER] Found full product data from catalog for {model}")
                        return extract_product_resources(full_product)
                except Exception as e:
                    logger.warning(f"[RESOURCE_LINKER] Failed to lookup product from catalog: {e}")
    
    # Try source_products (visual/catalog matches)
    source_products = state.get("source_products", [])
    if source_products:
        logger.info(f"[RESOURCE_LINKER] Checking {len(source_products)} source_products...")
        for product in source_products:
            if isinstance(product, dict):
                # Look for high-confidence match with URLs
                confidence = product.get("similarity_score", 0) or product.get("confidence", 0)
                if confidence >= 70 or product.get("match_level") == "🟢":
                    resources = extract_product_resources(product)
                    if resources.has_documents():
                        return resources
    
    # Try gathered_documents (from ReACT)
    gathered_docs = state.get("gathered_documents", [])
    if gathered_docs:
        # Look for product-related docs that might have model info
        for doc in gathered_docs:
            if isinstance(doc, dict) and doc.get("model_no"):
                return extract_product_resources(doc)
    
    logger.info("[RESOURCE_LINKER] No product with resource URLs found in state")
    return None


def build_resource_links_section(state: Dict[str, Any]) -> str:
    """
    Build HTML section with product resource links for response.
    
    This is the main entry point for draft_response.py.
    
    Args:
        state: Workflow state dictionary
        
    Returns:
        HTML string with resource links, or empty string if none found
    """
    resources = get_resources_from_state(state)
    
    if resources:
        logger.info(f"[RESOURCE_LINKER] Found resources for {resources.model_no}: "
                   f"docs={resources.has_documents()}, videos={resources.has_videos()}")
        return format_resources_html(resources)
    
    logger.debug("[RESOURCE_LINKER] No product resources found in state")
    return ""


def get_resource_context_for_llm(state: Dict[str, Any]) -> str:
    """
    Get plain text resource links for inclusion in LLM context.
    
    This helps the LLM reference specific documents in its response.
    
    Args:
        state: Workflow state dictionary
        
    Returns:
        Plain text string with resource links
    """
    resources = get_resources_from_state(state)
    
    if resources:
        return format_resources_text(resources)
    
    return ""
