"""
Resource Links Service
Fetches and formats product resource links (spec sheets, manuals, diagrams, videos)
for inclusion in Freshdesk responses.

This service uses the identified product's model number to fetch FULL product data
from the in-memory product catalog, ensuring accurate resource links.

IMPORTANT: Only call this service when product_confidence >= RESOURCE_LINKS_MIN_CONFIDENCE
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Minimum confidence threshold to include resource links
RESOURCE_LINKS_MIN_CONFIDENCE = 0.70


@dataclass
class ProductResources:
    """Container for validated product resource links."""
    
    model_no: str
    product_title: str = ""
    
    # Documents (PDFs)
    spec_sheet_url: Optional[str] = None
    install_manual_url: Optional[str] = None
    parts_diagram_url: Optional[str] = None
    
    # Videos
    install_video_url: Optional[str] = None
    operational_video_url: Optional[str] = None
    lifestyle_video_url: Optional[str] = None
    
    # Product page
    product_page_url: Optional[str] = None
    image_url: Optional[str] = None
    
    def has_any_resources(self) -> bool:
        """Check if any resources are available."""
        return any([
            self.spec_sheet_url,
            self.install_manual_url,
            self.parts_diagram_url,
            self.install_video_url,
            self.operational_video_url,
            self.lifestyle_video_url,
            self.product_page_url
        ])
    
    def has_documents(self) -> bool:
        """Check if any document links are available."""
        return any([self.spec_sheet_url, self.install_manual_url, self.parts_diagram_url])
    
    def has_videos(self) -> bool:
        """Check if any video links are available."""
        return any([self.install_video_url, self.operational_video_url, self.lifestyle_video_url])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_no": self.model_no,
            "product_title": self.product_title,
            "documents": {
                "spec_sheet": self.spec_sheet_url,
                "install_manual": self.install_manual_url,
                "parts_diagram": self.parts_diagram_url,
            },
            "videos": {
                "installation": self.install_video_url,
                "operational": self.operational_video_url,
                "lifestyle": self.lifestyle_video_url,
            },
            "product_page": self.product_page_url,
            "image": self.image_url,
        }


def _validate_url(url: str) -> Optional[str]:
    """
    Validate and normalize a URL.
    
    Returns:
        Normalized URL if valid, None otherwise
    """
    if not url:
        return None
    
    url = str(url).strip()
    
    # Skip empty or placeholder values
    if not url or url.lower() in ['', 'nan', 'none', 'n/a', '#']:
        return None
    
    # Ensure https:// prefix
    if not url.startswith(('http://', 'https://')):
        url = f"https://{url}"
    
    # Basic URL structure validation
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return None
        # Must have a valid domain
        if '.' not in parsed.netloc:
            return None
    except Exception:
        return None
    
    return url


def get_product_resources(model_no: str) -> Optional[ProductResources]:
    """
    Fetch complete product resources from the catalog using exact model number.
    
    This function queries the in-memory product catalog (already loaded at startup)
    to get the full product data including all resource URLs.
    
    Args:
        model_no: Exact product model number (e.g., "100.1170CP")
        
    Returns:
        ProductResources object with validated URLs, or None if product not found
    """
    if not model_no:
        logger.debug("[RESOURCE_LINKS] No model number provided")
        return None
    
    logger.info(f"[RESOURCE_LINKS] Fetching resources for model: {model_no}")
    
    try:
        # Import here to avoid circular imports
        from app.services.product_catalog import ensure_catalog_loaded
        
        catalog = ensure_catalog_loaded()
        
        # Try exact model search first
        product = catalog.search_exact_model(model_no)
        
        if not product:
            # Try group search (base model without finish)
            products = catalog.search_by_group(model_no)
            if products:
                product = products[0]
        
        if not product:
            logger.warning(f"[RESOURCE_LINKS] Product not found in catalog: {model_no}")
            return None
        
        # Extract and validate all resource URLs
        resources = ProductResources(
            model_no=product.get("model_no", model_no),
            product_title=product.get("title", ""),
            spec_sheet_url=_validate_url(product.get("spec_sheet_url", "")),
            install_manual_url=_validate_url(product.get("install_manual_url", "")),
            parts_diagram_url=_validate_url(product.get("parts_diagram_url", "")),
            install_video_url=_validate_url(product.get("install_video_url", "")),
            operational_video_url=_validate_url(product.get("operational_video_url", "")),
            lifestyle_video_url=_validate_url(product.get("lifestyle_video_url", "")),
            product_page_url=_validate_url(product.get("product_url", "")),
            image_url=_validate_url(product.get("image_url", "")),
        )
        
        if resources.has_any_resources():
            logger.info(f"[RESOURCE_LINKS] Found resources for {model_no}: "
                       f"docs={resources.has_documents()}, videos={resources.has_videos()}")
            return resources
        else:
            logger.info(f"[RESOURCE_LINKS] No valid resource URLs for {model_no}")
            return None
            
    except Exception as e:
        logger.error(f"[RESOURCE_LINKS] Error fetching resources for {model_no}: {e}")
        return None


def format_resources_html(resources: ProductResources) -> str:
    """
    Format product resources as HTML for inclusion in Freshdesk response.
    
    Generates a styled section with clickable buttons for each available resource.
    
    Args:
        resources: ProductResources object with validated URLs
        
    Returns:
        HTML string with resource links
    """
    if not resources or not resources.has_any_resources():
        return ""
    
    sections = []
    
    # === DOCUMENTS SECTION ===
    doc_links = []
    
    if resources.spec_sheet_url:
        doc_links.append(f'''
            <div style="display: flex; align-items: center; gap: 10px; margin: 6px 0;">
                <a href="{resources.spec_sheet_url}" target="_blank" rel="noopener noreferrer" 
                   style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; 
                          background: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 6px; 
                          text-decoration: none; color: #0369a1; font-size: 12px;
                          font-weight: 500; white-space: nowrap;">
                    📄 Spec Sheet
                </a>
                <code style="background: #f1f5f9; padding: 4px 8px; border-radius: 4px; font-size: 11px; color: #475569; word-break: break-all; max-width: 350px; overflow: hidden; text-overflow: ellipsis;">{resources.spec_sheet_url}</code>
            </div>''')
    
    if resources.install_manual_url:
        doc_links.append(f'''
            <div style="display: flex; align-items: center; gap: 10px; margin: 6px 0;">
                <a href="{resources.install_manual_url}" target="_blank" rel="noopener noreferrer" 
                   style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; 
                          background: #f0fdf4; border: 1px solid #22c55e; border-radius: 6px; 
                          text-decoration: none; color: #166534; font-size: 12px;
                          font-weight: 500; white-space: nowrap;">
                    📘 Installation Manual
                </a>
                <code style="background: #f1f5f9; padding: 4px 8px; border-radius: 4px; font-size: 11px; color: #475569; word-break: break-all; max-width: 350px; overflow: hidden; text-overflow: ellipsis;">{resources.install_manual_url}</code>
            </div>''')
    
    if resources.parts_diagram_url:
        doc_links.append(f'''
            <div style="display: flex; align-items: center; gap: 10px; margin: 6px 0;">
                <a href="{resources.parts_diagram_url}" target="_blank" rel="noopener noreferrer" 
                   style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; 
                          background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px; 
                          text-decoration: none; color: #92400e; font-size: 12px;
                          font-weight: 500; white-space: nowrap;">
                    🔧 Parts Diagram
                </a>
                <code style="background: #f1f5f9; padding: 4px 8px; border-radius: 4px; font-size: 11px; color: #475569; word-break: break-all; max-width: 350px; overflow: hidden; text-overflow: ellipsis;">{resources.parts_diagram_url}</code>
            </div>''')
    
    if doc_links:
        sections.append(f'''
        <div style="margin-bottom: 12px;">
            <div style="font-weight: 600; color: #374151; margin-bottom: 8px; font-size: 13px;">
                📁 Documents:
            </div>
            <div style="display: flex; flex-direction: column;">
                {''.join(doc_links)}
            </div>
        </div>''')
    
    # === VIDEOS SECTION ===
    video_links = []
    
    if resources.install_video_url:
        video_links.append(f'''
            <div style="display: flex; align-items: center; gap: 10px; margin: 6px 0;">
                <a href="{resources.install_video_url}" target="_blank" rel="noopener noreferrer" 
                   style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; 
                          background: #fef2f2; border: 1px solid #ef4444; border-radius: 6px; 
                          text-decoration: none; color: #dc2626; font-size: 12px;
                          font-weight: 500; white-space: nowrap;">
                    🎬 Installation Video
                </a>
                <code style="background: #f1f5f9; padding: 4px 8px; border-radius: 4px; font-size: 11px; color: #475569; word-break: break-all; max-width: 350px; overflow: hidden; text-overflow: ellipsis;">{resources.install_video_url}</code>
            </div>''')
    
    if resources.operational_video_url:
        video_links.append(f'''
            <div style="display: flex; align-items: center; gap: 10px; margin: 6px 0;">
                <a href="{resources.operational_video_url}" target="_blank" rel="noopener noreferrer" 
                   style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; 
                          background: #f5f3ff; border: 1px solid #8b5cf6; border-radius: 6px; 
                          text-decoration: none; color: #7c3aed; font-size: 12px;
                          font-weight: 500; white-space: nowrap;">
                    ▶️ Product Demo
                </a>
                <code style="background: #f1f5f9; padding: 4px 8px; border-radius: 4px; font-size: 11px; color: #475569; word-break: break-all; max-width: 350px; overflow: hidden; text-overflow: ellipsis;">{resources.operational_video_url}</code>
            </div>''')
    
    if resources.lifestyle_video_url:
        video_links.append(f'''
            <div style="display: flex; align-items: center; gap: 10px; margin: 6px 0;">
                <a href="{resources.lifestyle_video_url}" target="_blank" rel="noopener noreferrer" 
                   style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; 
                          background: #fdf4ff; border: 1px solid #d946ef; border-radius: 6px; 
                          text-decoration: none; color: #a21caf; font-size: 12px;
                          font-weight: 500; white-space: nowrap;">
                    🎥 Lifestyle Video
                </a>
                <code style="background: #f1f5f9; padding: 4px 8px; border-radius: 4px; font-size: 11px; color: #475569; word-break: break-all; max-width: 350px; overflow: hidden; text-overflow: ellipsis;">{resources.lifestyle_video_url}</code>
            </div>''')
    
    if video_links:
        sections.append(f'''
        <div style="margin-bottom: 12px;">
            <div style="font-weight: 600; color: #374151; margin-bottom: 8px; font-size: 13px;">
                🎬 Videos:
            </div>
            <div style="display: flex; flex-direction: column;">
                {''.join(video_links)}
            </div>
        </div>''')
    
    # === PRODUCT PAGE ===
    if resources.product_page_url:
        sections.append(f'''
        <div style="display: flex; align-items: center; gap: 10px; margin: 6px 0;">
            <a href="{resources.product_page_url}" target="_blank" rel="noopener noreferrer" 
               style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; 
                      background: #1e3a5f; border-radius: 6px; text-decoration: none; 
                      color: white; font-size: 12px; font-weight: 500; white-space: nowrap;">
                🌐 View Product Page
            </a>
            <code style="background: #f1f5f9; padding: 4px 8px; border-radius: 4px; font-size: 11px; color: #475569; word-break: break-all; max-width: 350px; overflow: hidden; text-overflow: ellipsis;">{resources.product_page_url}</code>
        </div>''')
    
    if not sections:
        return ""
    
    # Build title with model and product name
    title_text = f"📎 Quick Reference Links for {resources.model_no}"
    if resources.product_title:
        title_text = f"📎 Quick Reference Links: {resources.model_no} - {resources.product_title[:50]}"
    
    # Wrap in container
    html = f'''
    <div style="margin-top: 16px; padding: 16px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;">
        <div style="font-weight: bold; color: #1e3a5f; margin-bottom: 12px; font-size: 14px;">
            {title_text}
        </div>
        {''.join(sections)}
    </div>'''
    
    return html


def get_resource_links_for_response(
    identified_product: Optional[Dict[str, Any]],
    product_confidence: float
) -> str:
    """
    Main entry point for draft_response.py.
    
    Gets product resources and formats them as HTML if:
    1. Product was identified
    2. Confidence meets minimum threshold
    3. Product has valid resource URLs
    
    Args:
        identified_product: Product dict from ReACT agent (must have 'model' or 'model_no')
        product_confidence: Confidence score (0.0 to 1.0)
        
    Returns:
        HTML string with resource links, or empty string if not applicable
    """
    # Check confidence threshold
    if product_confidence < RESOURCE_LINKS_MIN_CONFIDENCE:
        logger.debug(f"[RESOURCE_LINKS] Skipping - confidence {product_confidence:.2f} < {RESOURCE_LINKS_MIN_CONFIDENCE}")
        return ""
    
    # Check if product was identified
    if not identified_product:
        logger.debug("[RESOURCE_LINKS] Skipping - no identified product")
        return ""
    
    # Get model number from identified_product
    # ReACT agent uses 'model', catalog uses 'model_no'
    model_no = (
        identified_product.get("model") or 
        identified_product.get("model_no") or 
        identified_product.get("Model_NO") or
        ""
    )
    
    if not model_no:
        logger.debug("[RESOURCE_LINKS] Skipping - no model number in identified product")
        return ""
    
    logger.info(f"[RESOURCE_LINKS] Processing product: {model_no} (confidence: {product_confidence:.2f})")
    
    # Fetch resources from catalog
    resources = get_product_resources(model_no)
    
    if not resources:
        return ""
    
    # Format as HTML
    return format_resources_html(resources)
