"""
Evidence Resolver Module
Handles conflict resolution when multiple sources provide different product identifications.
Implements smart decision logic for determining when to trust evidence vs request more info.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
STEP_NAME = "🔍 EVIDENCE_RESOLVER"

# ===============================
# THRESHOLDS (configurable)
# ===============================
VISION_HIGH_THRESHOLD = 0.90      # 90%+ = strong visual match
VISION_MEDIUM_THRESHOLD = 0.75   # 75-90% = possible match, needs corroboration
VISION_LOW_THRESHOLD = 0.75      # Below 75% = don't trust vision alone

OCR_CONFIDENCE_THRESHOLD = 0.80  # OCR must be 80%+ confident to trust


@dataclass
class EvidenceItem:
    """Single piece of evidence with source attribution"""
    source: str                    # "ocr", "vision", "product_search", "document_search", "past_tickets"
    product_model: Optional[str]   # Model number if identified
    product_name: Optional[str]    # Product name if identified
    product_category: Optional[str]
    confidence: float              # 0.0-1.0
    raw_data: Dict[str, Any] = field(default_factory=dict)
    is_exact_match: bool = False   # True if exact catalog match


@dataclass 
class EvidenceBundle:
    """Collection of all evidence with conflict analysis"""
    items: List[EvidenceItem] = field(default_factory=list)
    primary_product: Optional[Dict[str, Any]] = None
    has_conflict: bool = False
    conflict_reason: Optional[str] = None
    resolution_action: str = "proceed"  # "proceed", "request_info", "escalate"
    final_confidence: float = 0.0
    evidence_summary: str = ""


def categorize_vision_quality(similarity_score: float) -> str:
    """Categorize vision similarity into HIGH/MEDIUM/LOW"""
    if similarity_score >= VISION_HIGH_THRESHOLD:
        return "HIGH"
    elif similarity_score >= VISION_MEDIUM_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"


def analyze_evidence(
    ocr_result: Optional[Dict[str, Any]] = None,
    vision_result: Optional[Dict[str, Any]] = None,
    product_search_results: Optional[List[Dict[str, Any]]] = None,
    document_results: Optional[List[Dict[str, Any]]] = None,
    past_ticket_results: Optional[List[Dict[str, Any]]] = None,
    catalog_lookup_func = None,
    agent_identified_product: Optional[Dict[str, Any]] = None,
    agent_confidence: float = 0.0,
    ticket_facts: Optional[Dict[str, Any]] = None  # NEW: Pre-extracted ticket facts
) -> EvidenceBundle:
    """
    Analyze all evidence sources and determine best product identification.
    
    NOW INCLUDES ticket_facts as an evidence source to prevent the "Split Brain" problem
    where the agent knows info exists but evidence_resolver doesn't.
    
    Priority order:
    1. OCR exact match (explicit text, highest trust)
    2. ticket_facts verified models (from planner verification)
    3. Vision HIGH + corroborating evidence
    4. Vision HIGH alone (proceed with caution flag)
    5. Vision MEDIUM + corroborating evidence
    5. Exact catalog match from product search
    6. If none of above → request more information
    
    Returns:
        EvidenceBundle with analysis and recommendation
    """
    bundle = EvidenceBundle()
    evidence_items = []
    
    logger.info(f"{STEP_NAME} | Analyzing evidence from multiple sources...")
    
    # 1. Process OCR results
    if ocr_result and ocr_result.get("model_numbers"):
        models = ocr_result.get("model_numbers", [])
        for model in models:
            item = EvidenceItem(
                source="ocr",
                product_model=model,
                product_name=None,
                product_category=None,
                confidence=ocr_result.get("confidence", 0.9),
                raw_data=ocr_result,
                is_exact_match=True  # OCR is considered exact
            )
            evidence_items.append(item)
            logger.info(f"{STEP_NAME} | OCR found model: {model}")
    
    # 1.5. NEW: Process ticket_facts (pre-extracted from ticket text)
    # This closes the "Split Brain" gap - evidence_resolver now knows what ticket_extractor found
    ticket_facts_models = []
    if ticket_facts:
        raw_codes = ticket_facts.get("raw_product_codes", [])
        # NOTE: Key is planner_verified_models, not verified_models!
        verified_models = ticket_facts.get("planner_verified_models", [])
        
        # Use verified_models if available (planner-verified), else use raw codes
        # verified_models is List[str], raw_codes is List[dict]
        models_to_use = verified_models if verified_models else raw_codes
        
        for model_code in models_to_use:
            # Handle both dict format (from ticket_extractor) and string format
            if isinstance(model_code, dict):
                model_str = model_code.get("model") or model_code.get("full_sku", "")
                finish_code = model_code.get("finish_code")
                finish_name = model_code.get("finish_name")
            else:
                model_str = str(model_code)
                finish_code = None
                finish_name = None
            
            if not model_str:
                continue
                
            item = EvidenceItem(
                source="ticket_facts",
                product_model=model_str,
                product_name=None,
                product_category=None,
                confidence=0.85 if verified_models else 0.75,  # Higher if planner verified
                raw_data={
                    "source": "ticket_text", 
                    "verified": bool(verified_models),
                    "finish_code": finish_code,
                    "finish_name": finish_name
                },
                is_exact_match=True  # Customer explicitly mentioned this
            )
            evidence_items.append(item)
            ticket_facts_models.append(model_str)
            logger.info(f"{STEP_NAME} | Ticket facts model: {model_str} (verified={bool(verified_models)}, finish={finish_code})")
        
        # Log presence indicators
        has_model = ticket_facts.get("has_model_number", False)
        has_receipt = ticket_facts.get("has_receipt", False)
        has_address = ticket_facts.get("has_address", False)
        if has_model or has_receipt or has_address:
            logger.info(f"{STEP_NAME} | Ticket facts presence: model={has_model}, receipt={has_receipt}, address={has_address}")
    
    # 2. Process Vision results
    if vision_result and vision_result.get("matches"):
        matches = vision_result.get("matches", [])
        match_quality = vision_result.get("match_quality", "LOW")
        
        for match in matches[:3]:  # Top 3 vision matches
            similarity = match.get("similarity_score", 0) / 100 if match.get("similarity_score", 0) > 1 else match.get("similarity_score", 0)
            item = EvidenceItem(
                source="vision",
                product_model=match.get("model_no"),
                product_name=match.get("product_title"),
                product_category=match.get("category"),
                confidence=similarity,
                raw_data=match,
                is_exact_match=False
            )
            evidence_items.append(item)
            logger.info(f"{STEP_NAME} | Vision match: {match.get('model_no')} ({similarity:.1%})")
    
    # 3. Process Product Search results
    if product_search_results:
        logger.info(f"{STEP_NAME} | Processing {len(product_search_results)} product search result(s)")
        for result in product_search_results[:5]:  # Increased from 3 to 5
            # Check if it's an exact cache match vs semantic fallback
            source = result.get("source", "unknown")
            is_exact = source in ["catalog_cache", "exact", "group"] or result.get("exact_match", False)
            similarity = result.get("similarity_score", 50) / 100 if result.get("similarity_score", 0) > 1 else result.get("similarity_score", 0.5)
            
            item = EvidenceItem(
                source="product_search",
                product_model=result.get("model_no"),
                product_name=result.get("product_title"),
                product_category=result.get("category"),
                confidence=similarity if not is_exact else 0.95,
                raw_data=result,
                is_exact_match=is_exact
            )
            evidence_items.append(item)
            source_type = "exact" if is_exact else "semantic"
            logger.info(f"{STEP_NAME} | Product search ({source_type}): {result.get('model_no')} (source={source})")
    
    # 4. Check for document corroboration
    doc_mentioned_models = set()
    if document_results:
        for doc in document_results:
            # Extract model numbers mentioned in docs
            title = doc.get("title", "") or ""
            content = doc.get("content_preview", "") or doc.get("content", "") or ""
            
            # Simple extraction - look for model patterns in document titles/content
            import re
            # Improved patterns for Flusso model numbers
            patterns = [
                r'\b(\d{3}\.\d{4}[A-Z]{0,3})\b',    # 100.1050SB, 100.2420NB
                r'\b(\d{3}\.\d{3,4})[_\-\s]',       # 100.2420_ from filenames like 100.2420_253.pdf
                r'[/\\](\d{3}\.\d{3,4}[A-Z]{0,3})', # Extract from paths
                r'\b(\d{3}\.\d+[A-Z]*)\b',          # 160.6CSA (fallback)
            ]
            
            # First, try to extract from filename patterns like "100.2420_253.pdf"
            filename_pattern = r'(\d{3}\.\d{3,4}[A-Z]{0,3})(?:_\d+)?\.pdf'
            filename_match = re.search(filename_pattern, title, re.IGNORECASE)
            if filename_match:
                doc_mentioned_models.add(filename_match.group(1).upper())
            
            # Then try other patterns
            for pattern in patterns:
                found = re.findall(pattern, title + " " + content, re.IGNORECASE)
                doc_mentioned_models.update([m.upper() for m in found])
    
    if doc_mentioned_models:
        logger.info(f"{STEP_NAME} | Documents mention models: {doc_mentioned_models}")
    
    # ===============================
    # 5. PROCESS AGENT'S IDENTIFIED PRODUCT (CRITICAL FIX)
    # ===============================
    # The agent finds products via product_catalog_tool (exact matches).
    # We need to trust the agent's identification when:
    # - Confidence is high (>=0.70)
    # - Product was found via exact catalog match
    # - There's ANY related document (not necessarily exact model match)
    #
    # IMPORTANT: Document model numbers like 160.2420, 100.2420 ARE related to DKM.2420
    # (they're all in the same product family/category)
    # ===============================
    
    # ===============================
    # 5a. NEW: Handle "Product Inquiry" scenarios
    # ===============================
    # When customer mentions specific products AND documents were found for those products,
    # trust the agent's analysis even if we can't "identify" the product visually.
    # This handles cases like:
    # - "Can I upgrade PBV1005 to PBV2105 by changing the cartridge?"
    # - "What's the difference between model A and model B?"
    # The customer already told us the products - we just need to answer their question.
    # ===============================
    
    if document_results and agent_identified_product and agent_confidence >= 0.60:
        # Check if document_search found relevant specs (not just generic docs)
        found_product_specs = False
        docs_with_product_info = 0
        product_related_docs = []
        
        agent_model = agent_identified_product.get("model", "")
        
        for doc in document_results:
            doc_content = (doc.get("content_preview", "") or doc.get("content", "") or "").lower()
            doc_title = (doc.get("title", "") or "").lower()
            doc_score = doc.get("relevance_score", 0) or doc.get("score", 0)
            
            # Check if doc contains product specifications
            spec_indicators = [
                "specifications", "spec", "output", "diverter", "flow rate",
                "gpm", "psi", "dimensions", "valve body", "ports", "cartridge",
                "handle", "installation", "features", "warranty"
            ]
            has_specs = any(indicator in doc_content or indicator in doc_title for indicator in spec_indicators)
            
            # Check if doc mentions specific models (from agent or similar)
            if has_specs and doc_score >= 0.75:
                docs_with_product_info += 1
                product_related_docs.append(doc.get("title", "Unknown"))
        
        # If we found 2+ documents with product specs, trust the agent's analysis
        if docs_with_product_info >= 2:
            found_product_specs = True
            logger.info(f"{STEP_NAME} | ✅ Product inquiry mode: Found {docs_with_product_info} spec documents")
            logger.info(f"{STEP_NAME} | Spec docs: {product_related_docs[:3]}")
            
            bundle.primary_product = {
                "model": agent_model,
                "name": agent_identified_product.get("name", ""),
                "category": agent_identified_product.get("category", ""),
                "source": "product_inquiry_docs",
                "confidence": max(agent_confidence, 0.75)  # Boost confidence for product inquiries with docs
            }
            bundle.resolution_action = "proceed"
            bundle.final_confidence = max(agent_confidence, 0.75)
            bundle.evidence_summary = f"Product inquiry: Agent found {docs_with_product_info} specification documents for {agent_model}. Trust agent's technical analysis."
            
            logger.info(f"{STEP_NAME} | ✅ PRODUCT INQUIRY PATH: Trusting agent analysis with {docs_with_product_info} spec docs")
            return bundle
    
    agent_product_trusted = False
    if agent_identified_product and agent_confidence >= 0.70:
        agent_model = agent_identified_product.get("model", "")
        agent_name = agent_identified_product.get("name", "")
        agent_category = agent_identified_product.get("category", "")
        
        logger.info(f"{STEP_NAME} | Agent identified: {agent_model} ({agent_name}) with {agent_confidence:.0%} confidence")
        
        # Extract the numeric base from agent's model (e.g., "DKM.2420" -> "2420")
        import re
        agent_numeric_match = re.search(r'(\d{3,4})', agent_model)
        agent_numeric = agent_numeric_match.group(1) if agent_numeric_match else ""
        
        # Check multiple ways to corroborate:
        corroboration_found = False
        corroboration_reason = ""
        
        # Method 1: Direct document match (exact model in filename)
        for doc_model in doc_mentioned_models:
            if agent_model.upper() in doc_model or doc_model in agent_model.upper():
                corroboration_found = True
                corroboration_reason = f"Direct document match: {doc_model}"
                break
        
        # Method 2: Same product family (shared numeric base like "2420")
        if not corroboration_found and agent_numeric:
            for doc_model in doc_mentioned_models:
                if agent_numeric in doc_model:
                    corroboration_found = True
                    corroboration_reason = f"Related product family: {doc_model} shares {agent_numeric}"
                    break
        
        # Method 3: Product catalog exact match is trusted (source=exact or source=group)
        # Check if product_search_results has an exact match for the agent's model
        if not corroboration_found and product_search_results:
            for result in product_search_results:
                result_model = (result.get("model_no") or "").upper()
                is_exact = result.get("exact_match", False) or result.get("source") in ["exact", "group", "catalog_cache"]
                
                if is_exact and (result_model == agent_model.upper() or agent_model.upper() in result_model):
                    corroboration_found = True
                    corroboration_reason = f"Exact catalog match: {result_model}"
                    break
        
        # Method 4: High agent confidence alone (>=0.85) with documents found
        if not corroboration_found and agent_confidence >= 0.85 and document_results:
            corroboration_found = True
            corroboration_reason = f"High confidence ({agent_confidence:.0%}) with {len(document_results)} related documents"
        
        if corroboration_found:
            agent_product_trusted = True
            logger.info(f"{STEP_NAME} | ✅ Agent product trusted: {corroboration_reason}")
            
            item = EvidenceItem(
                source="agent_document_analysis",
                product_model=agent_model,
                product_name=agent_name,
                product_category=agent_category,
                confidence=agent_confidence,
                raw_data=agent_identified_product,
                is_exact_match=True  # Agent + corroboration = high trust
            )
            evidence_items.append(item)
        else:
            logger.info(f"{STEP_NAME} | ⚠️ Agent product not corroborated: {agent_model}")
    
    bundle.items = evidence_items
    
    # ===============================
    # CONFLICT RESOLUTION LOGIC (SIMPLIFIED)
    # ===============================
    
    if not evidence_items:
        bundle.resolution_action = "request_info"
        bundle.conflict_reason = "No product identification evidence found"
        bundle.final_confidence = 0.0
        bundle.evidence_summary = "No OCR, vision, or product search results available"
        logger.info(f"{STEP_NAME} | No evidence found → request more info")
        return bundle
    
    # Group evidence by product model (normalized)
    product_groups: Dict[str, List[EvidenceItem]] = {}
    for item in evidence_items:
        if item.product_model:
            normalized = item.product_model.upper().replace("_", ".").replace("-", ".")
            if normalized not in product_groups:
                product_groups[normalized] = []
            product_groups[normalized].append(item)
    
    logger.info(f"{STEP_NAME} | Found {len(product_groups)} distinct product(s) in evidence")
    
    # ===============================
    # PRIORITY 1: Agent's trusted product (if found)
    # ===============================
    if agent_product_trusted and agent_identified_product:
        bundle.primary_product = {
            "model": agent_identified_product.get("model"),
            "name": agent_identified_product.get("name"),
            "category": agent_identified_product.get("category"),
            "source": "agent_trusted",
            "confidence": agent_confidence
        }
        bundle.resolution_action = "proceed"
        bundle.final_confidence = agent_confidence
        bundle.evidence_summary = f"Agent identified product {agent_identified_product.get('model')} with {agent_confidence:.0%} confidence (trusted)"
        logger.info(f"{STEP_NAME} | ✅ Using agent's trusted product: {agent_identified_product.get('model')}")
        return bundle
    
    # ===============================
    # PRIORITY 2: OCR strong match 
    # ===============================
    ocr_items = [i for i in evidence_items if i.source == "ocr" and i.confidence >= OCR_CONFIDENCE_THRESHOLD]
    if ocr_items:
        best_ocr = max(ocr_items, key=lambda x: x.confidence)
        bundle.primary_product = {
            "model": best_ocr.product_model,
            "name": best_ocr.product_name,
            "category": best_ocr.product_category,
            "source": "ocr",
            "confidence": best_ocr.confidence
        }
        bundle.resolution_action = "proceed"
        bundle.final_confidence = 0.95
        bundle.evidence_summary = f"OCR extracted model {best_ocr.product_model} with high confidence"
        logger.info(f"{STEP_NAME} | ✅ OCR strong match: {best_ocr.product_model}")
        return bundle
    
    # ===============================
    # PRIORITY 2.5: Ticket Facts models (customer-provided in ticket text)
    # ===============================
    # This prevents asking for model numbers the customer already gave us
    ticket_facts_items = [i for i in evidence_items if i.source == "ticket_facts" and i.confidence >= 0.75]
    if ticket_facts_items:
        best_ticket_fact = max(ticket_facts_items, key=lambda x: x.confidence)
        bundle.primary_product = {
            "model": best_ticket_fact.product_model,
            "name": best_ticket_fact.product_name,
            "category": best_ticket_fact.product_category,
            "source": "ticket_facts",
            "confidence": best_ticket_fact.confidence
        }
        bundle.resolution_action = "proceed"
        bundle.final_confidence = 0.85
        verified_str = " (planner verified)" if best_ticket_fact.raw_data.get("verified") else ""
        bundle.evidence_summary = f"Customer provided model {best_ticket_fact.product_model} in ticket text{verified_str}"
        logger.info(f"{STEP_NAME} | ✅ Ticket facts match: {best_ticket_fact.product_model}")
        return bundle
    
    # Check for agent's document-based product identification (second highest priority)
    agent_doc_items = [i for i in evidence_items if i.source == "agent_document_analysis" and i.confidence >= 0.70]
    if agent_doc_items:
        best_agent = max(agent_doc_items, key=lambda x: x.confidence)
        bundle.primary_product = {
            "model": best_agent.product_model,
            "name": best_agent.product_name,
            "category": best_agent.product_category,
            "source": "agent_document_analysis",
            "confidence": best_agent.confidence
        }
        bundle.resolution_action = "proceed"
        bundle.final_confidence = 0.85
        bundle.evidence_summary = f"Agent identified product {best_agent.product_model} from document analysis with {best_agent.confidence:.0%} confidence"
        logger.info(f"{STEP_NAME} | ✅ Agent document analysis: {best_agent.product_model}")
        return bundle
    
    # Check vision results
    vision_items = [i for i in evidence_items if i.source == "vision"]
    best_vision = max(vision_items, key=lambda x: x.confidence) if vision_items else None
    
    if best_vision:
        vision_quality = categorize_vision_quality(best_vision.confidence)
        
        if vision_quality == "HIGH":
            # Check if vision product has corroborating evidence
            vision_model_norm = best_vision.product_model.upper().replace("_", ".") if best_vision.product_model else ""
            
            # Look for exact product search match
            exact_product_matches = [i for i in evidence_items 
                                     if i.source == "product_search" 
                                     and i.is_exact_match 
                                     and i.product_model 
                                     and i.product_model.upper().replace("_", ".") == vision_model_norm]
            
            # Check if docs mention this model
            has_doc_corroboration = vision_model_norm in doc_mentioned_models
            
            if exact_product_matches or has_doc_corroboration:
                # Vision HIGH + corroboration = ACCEPT
                bundle.primary_product = {
                    "model": best_vision.product_model,
                    "name": best_vision.product_name,
                    "category": best_vision.product_category,
                    "source": "vision_corroborated",
                    "confidence": best_vision.confidence
                }
                bundle.resolution_action = "proceed"
                bundle.final_confidence = 0.90
                bundle.evidence_summary = f"Vision HIGH ({best_vision.confidence:.0%}) + {'catalog' if exact_product_matches else 'document'} corroboration"
                logger.info(f"{STEP_NAME} | ✅ Vision HIGH with corroboration: {best_vision.product_model}")
                return bundle
            else:
                # Vision HIGH but no corroboration - proceed with caution
                bundle.primary_product = {
                    "model": best_vision.product_model,
                    "name": best_vision.product_name,
                    "category": best_vision.product_category,
                    "source": "vision_unverified",
                    "confidence": best_vision.confidence
                }
                bundle.has_conflict = True
                bundle.conflict_reason = "High visual match but no textual/catalog verification"
                bundle.resolution_action = "proceed_with_warning"
                bundle.final_confidence = 0.70
                bundle.evidence_summary = f"Vision HIGH ({best_vision.confidence:.0%}) but unverified - may need customer confirmation"
                logger.info(f"{STEP_NAME} | ⚠️ Vision HIGH unverified: {best_vision.product_model}")
                return bundle
        
        elif vision_quality == "MEDIUM":
            # Medium vision - require corroboration
            vision_model_norm = best_vision.product_model.upper().replace("_", ".") if best_vision.product_model else ""
            
            exact_matches = [i for i in evidence_items 
                           if i.source == "product_search" 
                           and i.is_exact_match 
                           and i.product_model]
            
            has_doc_corroboration = vision_model_norm in doc_mentioned_models
            
            if exact_matches or has_doc_corroboration:
                # Use the exact match if available
                if exact_matches:
                    best_exact = exact_matches[0]
                    bundle.primary_product = {
                        "model": best_exact.product_model,
                        "name": best_exact.product_name,
                        "category": best_exact.product_category,
                        "source": "product_search_exact",
                        "confidence": best_exact.confidence
                    }
                else:
                    bundle.primary_product = {
                        "model": best_vision.product_model,
                        "name": best_vision.product_name,
                        "category": best_vision.product_category,
                        "source": "vision_doc_corroborated",
                        "confidence": best_vision.confidence
                    }
                bundle.resolution_action = "proceed"
                bundle.final_confidence = 0.75
                bundle.evidence_summary = f"Vision MEDIUM ({best_vision.confidence:.0%}) with corroboration"
                logger.info(f"{STEP_NAME} | ✅ Vision MEDIUM with corroboration")
                return bundle
            else:
                # Medium vision, no corroboration - request info
                bundle.resolution_action = "request_info"
                bundle.conflict_reason = "Medium visual match without verification"
                bundle.final_confidence = 0.40
                bundle.evidence_summary = f"Vision MEDIUM ({best_vision.confidence:.0%}) but no supporting evidence"
                logger.info(f"{STEP_NAME} | ⚠️ Vision MEDIUM without corroboration → request info")
                return bundle
        
        else:
            # Low vision - always request info unless we have exact match
            exact_matches = [i for i in evidence_items 
                           if i.source == "product_search" 
                           and i.is_exact_match]
            
            if exact_matches:
                best_exact = exact_matches[0]
                bundle.primary_product = {
                    "model": best_exact.product_model,
                    "name": best_exact.product_name,
                    "category": best_exact.product_category,
                    "source": "product_search_exact",
                    "confidence": best_exact.confidence
                }
                bundle.resolution_action = "proceed_with_warning"
                bundle.final_confidence = 0.60
                bundle.evidence_summary = f"Vision LOW but exact catalog match found"
                return bundle
            
            bundle.resolution_action = "request_info"
            bundle.conflict_reason = "Low visual match confidence"
            bundle.final_confidence = 0.25
            bundle.evidence_summary = f"Vision LOW ({best_vision.confidence:.0%}) - cannot reliably identify product"
            logger.info(f"{STEP_NAME} | ❌ Vision LOW → request info")
            return bundle
    
    # No vision results - check product search
    exact_matches = [i for i in evidence_items if i.source == "product_search" and i.is_exact_match]
    if exact_matches:
        best_exact = exact_matches[0]
        bundle.primary_product = {
            "model": best_exact.product_model,
            "name": best_exact.product_name,
            "category": best_exact.product_category,
            "source": "product_search_exact",
            "confidence": best_exact.confidence
        }
        bundle.resolution_action = "proceed"
        bundle.final_confidence = 0.85
        bundle.evidence_summary = f"Exact catalog match: {best_exact.product_model}"
        return bundle
    
    # Fallback - no strong evidence
    bundle.resolution_action = "request_info"
    bundle.conflict_reason = "No reliable product identification found"
    bundle.final_confidence = 0.20
    bundle.evidence_summary = "Insufficient evidence to identify product"
    logger.info(f"{STEP_NAME} | ❌ No strong evidence → request info")
    
    return bundle


def detect_evidence_conflicts(evidence_items: List[EvidenceItem]) -> Tuple[bool, Optional[str]]:
    """
    Detect if evidence sources conflict with each other.
    
    Returns:
        (has_conflict: bool, conflict_description: str or None)
    """
    # Group by source
    by_source: Dict[str, List[EvidenceItem]] = {}
    for item in evidence_items:
        if item.source not in by_source:
            by_source[item.source] = []
        by_source[item.source].append(item)
    
    # Get top product from vision vs product_search
    vision_top = None
    product_search_top = None
    
    if "vision" in by_source and by_source["vision"]:
        vision_top = max(by_source["vision"], key=lambda x: x.confidence)
    
    if "product_search" in by_source and by_source["product_search"]:
        product_search_top = max(by_source["product_search"], key=lambda x: x.confidence)
    
    if vision_top and product_search_top:
        # Normalize and compare
        v_model = (vision_top.product_model or "").upper().replace("_", ".").replace("-", ".")
        p_model = (product_search_top.product_model or "").upper().replace("_", ".").replace("-", ".")
        
        if v_model and p_model and v_model != p_model:
            # Check if they're in different categories
            v_cat = (vision_top.product_category or "").lower()
            p_cat = (product_search_top.product_category or "").lower()
            
            if v_cat and p_cat and v_cat != p_cat:
                return True, f"Category mismatch: Vision identified '{v_cat}' ({v_model}) but product search found '{p_cat}' ({p_model})"
            else:
                return True, f"Model mismatch: Vision identified {v_model} but product search found {p_model}"
    
    return False, None


def generate_info_request_response(
    bundle: EvidenceBundle,
    customer_name: str = "Customer",
    ticket_subject: str = "",
    ticket_text: str = "",
    ticket_category: str = "",
    ticket_facts: Optional[Dict[str, Any]] = None  # NEW: Pre-extracted facts
) -> Dict[str, str]:
    """
    Generate info request response - returns empty to force LLM-generated response.
    
    NOW CHECKS ticket_facts before deciding what info to request.
    If ticket_facts has model numbers, we should NOT ask for model numbers again.
    
    NOTE: We no longer use hardcoded placeholder messages like:
    "To locate the correct replacement part, please provide model number..."
    
    This caused issues where the agent had already gathered relevant specs but
    the placeholder message was sent anyway (see ticket #97841 analysis).
    
    Instead, return empty so the workflow uses the LLM-generated response
    based on actual gathered context.
    
    Returns:
        {
            "customer_message": "" (empty - forces LLM generation),
            "private_note": str (summary for human agent)
        }
    """
    # Build private note for human agent review
    possible_product = bundle.primary_product.get("model") if bundle.primary_product else None
    
    # Check what info we already have from ticket_facts
    has_model_from_facts = False
    has_receipt_from_facts = False
    has_address_from_facts = False
    known_models = []
    
    if ticket_facts:
        raw_codes = ticket_facts.get("raw_product_codes", [])
        # NOTE: Key is planner_verified_models, not verified_models!
        verified_models = ticket_facts.get("planner_verified_models", [])
        
        # verified_models is List[str], raw_codes is List[dict]
        # We need to normalize to List[str] for display
        if verified_models:
            known_models = verified_models  # Already strings
        else:
            # Extract model strings from raw_codes dicts
            known_models = [c.get("model") for c in raw_codes if isinstance(c, dict) and c.get("model")]
        
        has_model_from_facts = bool(known_models)
        has_receipt_from_facts = ticket_facts.get("has_receipt", False)
        has_address_from_facts = ticket_facts.get("has_address", False)
        
        if known_models:
            logger.info(f"{STEP_NAME} | generate_info_request: Customer already provided models: {known_models}")
    
    # Detect request type for private note
    ticket_lower = (ticket_text + " " + ticket_subject).lower()
    is_parts_request = any(word in ticket_lower for word in ["part", "replacement", "spare", "broken", "repair", "fix", "cartridge", "diverter"])
    is_warranty_request = any(word in ticket_lower for word in ["warranty", "defect", "issue", "problem", "leak"])
    is_return_request = any(word in ticket_lower for word in ["return", "refund", "exchange", "send back"])
    request_type = "Parts" if is_parts_request else "Warranty" if is_warranty_request else "Return" if is_return_request else "General"
    
    # Build info status for private note
    info_status = []
    if has_model_from_facts:
        info_status.append(f"✅ Model(s) provided: {', '.join(known_models)}")
    else:
        info_status.append("❓ Model number: Not found in ticket")
    
    if has_receipt_from_facts:
        info_status.append("✅ Receipt: Present")
    if has_address_from_facts:
        info_status.append("✅ Address: Present")
    request_type = "Parts" if is_parts_request else "Warranty" if is_warranty_request else "Return" if is_return_request else "General"
    
    # Build info status for private note
    info_status = []
    if has_model_from_facts:
        info_status.append(f"✅ Model(s) provided: {', '.join(known_models)}")
    else:
        info_status.append("❓ Model number: Not found in ticket")
    
    if has_receipt_from_facts:
        info_status.append("✅ Receipt: Present")
    if has_address_from_facts:
        info_status.append("✅ Address: Present")
    
    info_status_str = "\n".join(info_status)
    
    private_note = f"""🤖 **AI Summary**
• Request: {request_type}
• Confidence: {int(bundle.final_confidence * 100)}%
• Best guess: {possible_product or 'Unknown'}
• Reason: {bundle.conflict_reason or 'Insufficient evidence for confident response'}

**Customer-Provided Info:**
{info_status_str}

**Action:** Review gathered documents and respond based on available information."""
    
    # Return empty customer_message - this forces the workflow to use 
    # the LLM-generated response based on actual gathered context
    return {
        "customer_message": "",
        "private_note": private_note
    }


def should_request_more_info(
    vision_quality: str,
    vision_confidence: float,
    has_ocr_result: bool,
    has_exact_catalog_match: bool,
    has_document_corroboration: bool
) -> Tuple[bool, str]:
    """
    Quick check if we should request more info from customer.
    
    Returns:
        (should_request: bool, reason: str)
    """
    # OCR found something = proceed
    if has_ocr_result:
        return False, "OCR found model number"
    
    # Vision HIGH + any corroboration = proceed
    if vision_confidence >= VISION_HIGH_THRESHOLD:
        if has_exact_catalog_match or has_document_corroboration:
            return False, "High vision match with corroboration"
        else:
            return False, "High vision match (proceed with caution)"
    
    # Vision MEDIUM + corroboration = proceed
    if vision_confidence >= VISION_MEDIUM_THRESHOLD:
        if has_exact_catalog_match or has_document_corroboration:
            return False, "Medium vision match with corroboration"
        else:
            return True, "Medium vision match without verification"
    
    # Vision LOW or no vision
    if has_exact_catalog_match:
        return False, "Exact catalog match found"
    
    return True, "Insufficient evidence for reliable identification"
