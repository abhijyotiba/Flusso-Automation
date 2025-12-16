"""
ReACT Agent Helper Functions - FIXED VERSION
Context building, tool execution, legacy field population
"""

import json
import logging
from typing import Dict, Any, List, Tuple, Optional

# =============================================================================
# TOOL IMPORTS
# =============================================================================
# NEW: Import from the comprehensive product catalog tool
from app.tools.product_catalog_tool import product_catalog_tool, get_product_variations

from app.tools.document_search import document_search_tool
from app.tools.vision_search import vision_search_tool
from app.tools.past_tickets import past_tickets_search_tool
from app.tools.finish import finish_tool
from app.tools.attachment_analyzer import attachment_analyzer_tool
from app.tools.attachment_classifier_tool import attachment_type_classifier_tool
from app.tools.multimodal_document_analyzer import multimodal_document_analyzer_tool
from app.tools.ocr_image_analyzer import ocr_image_analyzer_tool

logger = logging.getLogger(__name__)


def _build_agent_context(
    ticket_subject: str,
    ticket_text: str,
    ticket_images: List[str],
    attachments: List[Dict],
    iterations: List[Dict],
    tool_results: Dict[str, Any],
    iteration_num: int,
    max_iterations: int
) -> str:
    """Build context for ReACT agent with ticket info and history"""
    
    context_parts = [
        f"═══ ITERATION {iteration_num}/{max_iterations} ═══\n",
        f"CUSTOMER TICKET:",
        f"Subject: {ticket_subject}",
        f"\nDescription:\n{ticket_text[:2000]}",  # Limit to 2000 chars
    ]
    
    if ticket_images:
        context_parts.append(f"\n📷 Images Attached: {len(ticket_images)} image(s)")
        context_parts.append(f"Image URLs: {ticket_images}")
    
    if attachments:
        context_parts.append(f"\n📎 Documents Attached: {len(attachments)} file(s)")
        att_info = []
        for att in attachments[:5]:  # Show first 5
            att_info.append(f"  - {att.get('filename', 'Unknown')} ({att.get('type', 'unknown')})")
        context_parts.append("\n".join(att_info))
    
    # Show what tools have been used
    if iterations:
        context_parts.append(f"\n\n═══ PREVIOUS ACTIONS ═══")
        for it in iterations[-5:]:  # Show last 5
            context_parts.append(f"\nIteration {it['iteration']}:")
            context_parts.append(f"  Thought: {it['thought'][:150]}")
            context_parts.append(f"  Action: {it['action']}")
            # ⚠️ CRITICAL: Show MORE of the observation - model numbers were being truncated!
            # Increase from 200 to 1000 chars to ensure identifiers are preserved
            obs_preview = it['observation'][:1000]
            context_parts.append(f"  Result: {obs_preview}")
    
    # Show accumulated results
    context_parts.append(f"\n\n═══ CURRENT STATE ═══")
    
    if tool_results["product_search"]:
        prod = tool_results["product_search"]
        if prod.get("success"):
            context_parts.append(f"✓ Product Search: Found {prod.get('count', 0)} product(s)")
            if prod.get("products"):
                top = prod["products"][0]
                context_parts.append(f"  Top match: {top.get('model_no')} - {top.get('product_title')}")
                if prod.get("source") == "catalog_cache":
                    context_parts.append(f"  (Source: Verified Catalog Match)")
    
    if tool_results["document_search"]:
        docs = tool_results["document_search"]
        if docs.get("success"):
            context_parts.append(f"✓ Document Search: Found {docs.get('count', 0)} document(s)")
    
    if tool_results["vision_search"]:
        vis = tool_results["vision_search"]
        if vis.get("success"):
            context_parts.append(f"✓ Vision Search: Quality={vis.get('match_quality')}, {vis.get('count', 0)} match(es)")
    
    if tool_results["past_tickets"]:
        past = tool_results["past_tickets"]
        if past.get("success"):
            context_parts.append(f"✓ Past Tickets: Found {past.get('count', 0)} similar ticket(s)")
    
    if tool_results["attachment_analysis"]:
        att_analysis = tool_results["attachment_analysis"]
        if att_analysis.get("success"):
            extracted = att_analysis.get("extracted_info", {})
            models = extracted.get("model_numbers", [])
            if models:
                context_parts.append(f"✓ Attachment Analysis: Model numbers extracted: {models}")

        if tool_results.get("attachment_classification"):
            ac = tool_results["attachment_classification"]
            if ac.get("success"):
                types = [f"{a['name']}: {a['detected_type']}" for a in ac.get("attachments", [])]
                context_parts.append(f"✓ Attachment Classification: {', '.join(types)}")

        if tool_results.get("multimodal_doc_analysis"):
            mda = tool_results["multimodal_doc_analysis"]
            if mda.get("success"):
                context_parts.append(f"✓ Multimodal Doc Analysis: {mda.get('count', 0)} document(s) processed")
                
                # ⚠️ CRITICAL: Extract and prominently show ALL identifiers!
                all_ids = mda.get("all_identifiers", {})
                if all_ids:
                    model_nums = all_ids.get("model_numbers", [])
                    part_nums = all_ids.get("part_numbers", [])
                    order_nums = all_ids.get("order_numbers", [])
                    
                    if model_nums:
                        context_parts.append(f"  🔴 MODEL NUMBERS EXTRACTED: {', '.join(model_nums)}")
                        context_parts.append(f"     ➡️ Use these in product_catalog_tool!")
                    if part_nums:
                        context_parts.append(f"  🟠 PART NUMBERS EXTRACTED: {', '.join(part_nums)}")
                    if order_nums:
                        context_parts.append(f"  🟡 ORDER NUMBERS EXTRACTED: {', '.join(order_nums)}")
                
                # Also show per-document info
                for doc in mda.get("documents", [])[:2]:
                    doc_ids = doc.get("identifiers", {})
                    doc_models = doc_ids.get("model_numbers", [])
                    filename = doc.get('filename', 'unknown')
                    if doc_models:
                        context_parts.append(f"  📄 {filename}: models={doc_models}")

        if tool_results.get("ocr_image_analysis"):
            ocr = tool_results["ocr_image_analysis"]
            if ocr.get("success"):
                context_parts.append(f"✓ OCR Image Analysis: {len(ocr.get('results', []))} image(s) processed")
                
                # ⚠️ CRITICAL: Extract and prominently show ALL identifiers!
                all_ids = ocr.get("all_identifiers", {})
                if all_ids:
                    model_nums = all_ids.get("model_numbers", [])
                    part_nums = all_ids.get("part_numbers", [])
                    order_nums = all_ids.get("order_numbers", [])
                    
                    if model_nums:
                        context_parts.append(f"  🔴 MODEL NUMBERS EXTRACTED: {', '.join(model_nums)}")
                        context_parts.append(f"     ➡️ Use these in product_catalog_tool!")
                    if part_nums:
                        context_parts.append(f"  🟠 PART NUMBERS EXTRACTED: {', '.join(part_nums)}")
                    if order_nums:
                        context_parts.append(f"  🟡 ORDER NUMBERS EXTRACTED: {', '.join(order_nums)}")
                        context_parts.append(f"     (Order numbers are NOT model numbers)")
                
                # Show the full combined text (or first 800 chars) so agent can see model numbers
                combined_text = ocr.get("combined_text", "")
                if combined_text and combined_text.strip():
                    text_preview = combined_text[:800] if len(combined_text) > 800 else combined_text
                    context_parts.append(f"  📄 Extracted Text:\n{text_preview}")
                    
                    # Highlight any potential model numbers found (backup regex detection)
                    import re
                    model_patterns = [
                        r'\b(\d{2,3}\.\d{3,4}[A-Z]{0,3})\b',
                        r'\b([A-Z]{2,5}-?\d{3,6}[A-Z]{0,3})\b',
                    ]
                    found_models = []
                    for pattern in model_patterns:
                        matches = re.findall(pattern, combined_text, re.IGNORECASE)
                        found_models.extend(matches)
                    if found_models:
                        unique_models = list(set(found_models))[:5]
                        context_parts.append(f"  🔍 POTENTIAL MODEL NUMBERS: {', '.join(unique_models)}")
                else:
                    context_parts.append(f"  [No readable text extracted]")
    
    # Add urgency if approaching limit - make it VERY prominent
    if iteration_num >= max_iterations - 2:
        context_parts.append(f"\n\n{'='*60}")
        context_parts.append(f"🛑 CRITICAL URGENCY ALERT 🛑")
        context_parts.append(f"{'='*60}")
        context_parts.append(f"⚠️ Only {max_iterations - iteration_num} iteration(s) remaining!")
        context_parts.append(f"⚠️ You MUST call finish_tool NOW with whatever information you have!")
        context_parts.append(f"⚠️ Do NOT attempt any more searches - you're out of time!")
        context_parts.append(f"{'='*60}\n")
    elif iteration_num >= max_iterations - 3:
        context_parts.append(f"\n⚠️ WARNING: Only {max_iterations - iteration_num} iterations remaining!")
        context_parts.append("⚠️ Consider calling finish_tool soon with your current findings.")
    
    return "\n".join(context_parts)


def _execute_tool(
    action: str,
    action_input: Dict[str, Any],
    ticket_images: List[str],
    attachments: List[Dict],
    tool_results: Dict[str, Any],
    identified_product: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], str]:
    """
    Execute the chosen tool with proper parameter handling.
    """
    
    try:
        # Helper to run LangChain tools properly
        def _run_langchain_tool(tool, params: Dict[str, Any]) -> Any:
            """
            Standard LangChain tool execution.
            Always wraps params in 'tool_input' key.
            """
            try:
                # LangChain tools expect tool_input parameter
                return tool.run(tool_input=params)
            except TypeError:
                # Fallback: try direct invocation
                return tool.invoke(params)
        
        # -----------------------------------------------------------
        # AUTO-DETECT MODEL NUMBERS FOR PRODUCT SEARCH
        # -----------------------------------------------------------
        def _looks_like_model_number(text: Any) -> bool:
            """Check if text looks like a product model number"""
            if not isinstance(text, str):
                return False
            import re
            # Match patterns like: 100.1170, HS6270MB, 160.1168-9862
            return bool(re.match(r"^[A-Z0-9]{1,4}[\.\-]?[0-9]{3,5}[\-A-Z0-9]*$", text.strip(), re.I))
        
        # -----------------------------------------------------------
        # 1. ATTACHMENT ANALYZER TOOL
        # -----------------------------------------------------------
        if action == "attachment_analyzer_tool":
            logger.info(f"[TOOL_EXEC] Executing attachment_analyzer_tool")
            
            if not action_input:
                action_input = {}
            
            # Always pass attachments explicitly
            action_input["attachments"] = attachments
            
            # Set focus if not provided
            if "focus" not in action_input:
                action_input["focus"] = "model_numbers"
            
            output = _run_langchain_tool(attachment_analyzer_tool, action_input)
            tool_results["attachment_analysis"] = output
            
            if output.get("success"):
                extracted = output.get("extracted_info", {})
                models = extracted.get("model_numbers", [])
                obs = f"Analyzed {output.get('count', 0)} attachment(s). "
                if models:
                    obs += f"Found model numbers: {', '.join(models[:5])}"
                    if len(models) > 5:
                        obs += f" (and {len(models)-5} more)"
                else:
                    obs += "No model numbers extracted."
                return output, obs
            else:
                return output, f"Attachment analysis failed: {output.get('message')}"
        
        # -----------------------------------------------------------
        # 2. PRODUCT CATALOG TOOL (Comprehensive Product Search)
        # -----------------------------------------------------------
        elif action == "product_catalog_tool" or action == "product_search_tool":
            logger.info(f"[TOOL_EXEC] Executing product_catalog_tool")
            
            # Make action_input mutable
            action_input = dict(action_input or {})
            
            # Auto-fix: If query looks like model number, set model_number parameter
            query = action_input.get("query", "")
            if query and _looks_like_model_number(query) and "model_number" not in action_input:
                logger.info(f"[TOOL_EXEC] Auto-detected model number in query: {query}")
                action_input["model_number"] = query
                action_input.pop("query", None)  # Remove query param
            
            # Call the new comprehensive product catalog tool
            output = _run_langchain_tool(product_catalog_tool, action_input)
            tool_results["product_search"] = output
            
            if output.get("success"):
                count = output.get("count", 0)
                method = output.get("search_method", "unknown")
                products = output.get("products", [])
                variations = output.get("variations", {})
                
                # Build rich observation
                obs = f"✅ Product search successful ({method} match).\n\n"
                
                if products:
                    top = products[0]
                    obs += f"📦 PRODUCT: {top.get('model_no')}\n"
                    obs += f"   Title: {top.get('title')}\n"
                    obs += f"   Category: {top.get('category')} > {top.get('sub_category')}\n"
                    obs += f"   Collection: {top.get('collection')}\n"
                    obs += f"   Finish: {top.get('finish_name')} ({top.get('finish_code')})\n"
                    
                    # Pricing
                    price = top.get('list_price', 0)
                    if price:
                        obs += f"   Price: ${price:,.2f}\n"
                    
                    # Specifications
                    dims = top.get('dimensions', {})
                    if dims and any(dims.values()):
                        obs += f"   Dimensions: {dims.get('height')}\"H x {dims.get('length')}\"L x {dims.get('width')}\"W\n"
                    
                    flow = top.get('flow_rate_gpm', 0)
                    if flow:
                        obs += f"   Flow Rate: {flow} GPM\n"
                    
                    # Available resources
                    resources = []
                    if top.get('spec_sheet_url'):
                        resources.append("Spec Sheet")
                    if top.get('install_manual_url'):
                        resources.append("Install Manual")
                    if top.get('parts_diagram_url'):
                        resources.append("Parts Diagram")
                    if top.get('install_video_url'):
                        resources.append("Install Video")
                    if resources:
                        obs += f"   Resources: ✓ {', '.join(resources)}\n"
                    
                    # Features
                    features = top.get('features', [])
                    if features:
                        obs += f"   Features:\n"
                        for feat in features[:4]:
                            obs += f"     • {feat}\n"
                    
                    # Warranty
                    warranty = top.get('warranty')
                    if warranty:
                        obs += f"   Warranty: {warranty[:80]}...\n" if len(warranty) > 80 else f"   Warranty: {warranty}\n"
                
                # Show finish variations if available
                if variations and len(variations) > 1:
                    obs += f"\n🎨 AVAILABLE FINISHES ({len(variations)}):\n"
                    finish_list = [f"{code} ({name})" for code, name in variations.items()]
                    obs += f"   {', '.join(finish_list)}\n"
                
                # Group search note
                if method == "group" and count > 1:
                    obs += f"\n📋 NOTE: Found {count} finish variations for this product group.\n"
                
                # Fuzzy match suggestions
                suggestions = output.get("suggestions", [])
                if suggestions and method == "fuzzy":
                    obs += f"\n🔍 DID YOU MEAN:\n"
                    for sug in suggestions[:3]:
                        obs += f"   • {sug['model_no']} ({sug['similarity']}% match) - {sug['title']}\n"
                
                # Related spare parts
                related = output.get("related_parts", [])
                if related:
                    obs += f"\n🔧 RELATED SPARE PARTS:\n"
                    for part in related[:3]:
                        obs += f"   • {part['model_no']} - {part['title']}\n"
                
                logger.info(f"[TOOL_EXEC] Product search: {count} result(s) via {method}")
                return output, obs
            else:
                return output, f"❌ No products found: {output.get('message')}"
        
        # -----------------------------------------------------------
        # 2b. GET PRODUCT VARIATIONS (helper tool)
        # -----------------------------------------------------------
        elif action == "get_product_variations":
            logger.info(f"[TOOL_EXEC] Executing get_product_variations")
            
            output = _run_langchain_tool(get_product_variations, action_input)
            
            if output.get("success"):
                variations = output.get("variations", {})
                obs = f"Found {len(variations)} finish variation(s) for {output.get('group_number')}:\n"
                for code, info in variations.items():
                    obs += f"  • {info['model_no']} - {info['finish_name']} (${info['price']:,.2f})\n"
                return output, obs
            else:
                return output, f"No variations found: {output.get('message')}"
        
        # -----------------------------------------------------------
        # 3. DOCUMENT SEARCH TOOL
        # -----------------------------------------------------------
        elif action == "document_search_tool":
            logger.info(f"[TOOL_EXEC] Executing document_search_tool")
            
            action_input = dict(action_input or {})
            
            # Auto-add product context if available
            if identified_product and "product_context" not in action_input:
                model = identified_product.get("model")
                name = identified_product.get("name")
                if model or name:
                    action_input["product_context"] = model or name
                    logger.info(f"[TOOL_EXEC] Added product context: {action_input['product_context']}")
            
            
            output = _run_langchain_tool(document_search_tool, action_input)
            tool_results["document_search"] = output
            
            if output.get("success"):
                count = output.get("count", 0)
                obs = f"Found {count} document(s). "
                if output.get("documents"):
                    titles = [d.get("title", "Untitled") for d in output["documents"][:3]]
                    obs += f"Top docs: {', '.join(titles)}"
                return output, obs
            else:
                return output, f"No documents found: {output.get('message')}"
        
        # -----------------------------------------------------------
        # 4. VISION SEARCH TOOL
        # -----------------------------------------------------------
        elif action == "vision_search_tool":
            logger.info(f"[TOOL_EXEC] Executing vision_search_tool")
            
            action_input = dict(action_input or {})
            action_input["image_urls"] = ticket_images
            
            logger.info(f"[TOOL_EXEC] Searching {len(ticket_images)} image(s)")
            
            output = _run_langchain_tool(vision_search_tool, action_input)
            tool_results["vision_search"] = output
            
            if output.get("success"):
                quality = output.get("match_quality")
                count = output.get("count", 0)
                obs = f"Vision match quality: {quality}. Found {count} match(es). "
                obs += output.get("reasoning", "")
                return output, obs
            else:
                return output, f"Vision search failed: {output.get('message')}"
        
        # -----------------------------------------------------------
        # 5. PAST TICKETS SEARCH TOOL
        # -----------------------------------------------------------
        elif action == "past_tickets_search_tool":
            logger.info(f"[TOOL_EXEC] Executing past_tickets_search_tool")
            
            output = _run_langchain_tool(past_tickets_search_tool, action_input or {})
            tool_results["past_tickets"] = output
            
            if output.get("success"):
                count = output.get("count", 0)
                obs = f"Found {count} similar past ticket(s). "
                patterns = output.get("common_patterns", [])
                if patterns:
                    obs += f"Patterns: {'; '.join(patterns)}"
                return output, obs
            else:
                return output, f"No past tickets found: {output.get('message')}"
        
        # -----------------------------------------------------------
        # 6. ATTACHMENT TYPE CLASSIFIER TOOL
        # -----------------------------------------------------------
        elif action == "attachment_type_classifier_tool":
            logger.info(f"[TOOL_EXEC] Executing attachment_type_classifier_tool")
            
            action_input = dict(action_input or {})
            action_input["attachments"] = attachments
            
            output = _run_langchain_tool(attachment_type_classifier_tool, action_input)
            tool_results["attachment_classification"] = output
            
            if output.get("success"):
                types_list = [a.get('detected_type', 'unknown') for a in output.get('attachments', [])]
                obs = f"Attachment types classified: {types_list}"
                return output, obs
            else:
                return output, f"Attachment type classification failed: {output.get('message')}"
        
        # -----------------------------------------------------------
        # 7. MULTIMODAL DOCUMENT ANALYZER TOOL (Intelligent Document Analyzer)
        # -----------------------------------------------------------
        elif action == "multimodal_document_analyzer_tool":
            logger.info(f"[TOOL_EXEC] Executing multimodal_document_analyzer_tool (Intelligent Document Analyzer)")
            
            action_input = dict(action_input or {})
            action_input["attachments"] = attachments
            
            output = _run_langchain_tool(multimodal_document_analyzer_tool, action_input)
            tool_results["multimodal_doc_analysis"] = output
            
            if output.get("success"):
                docs = output.get("documents", [])
                doc_types = output.get("document_types", {})
                all_ids = output.get("all_identifiers", {})
                
                logger.info(f"[TOOL_EXEC] 📄 Document analyzer processed {len(docs)} document(s)")
                logger.info(f"[TOOL_EXEC] 📄 Document types: {doc_types}")
                
                # =====================================================
                # VERBOSE OUTPUT - Print full analysis to terminal
                # =====================================================
                print("\n" + "=" * 80)
                print("📄 MULTIMODAL DOCUMENT ANALYZER - FULL OUTPUT")
                print("=" * 80)
                
                for doc in docs:
                    filename = doc.get("filename", "unknown")
                    status = doc.get("status", "unknown")
                    doc_type = doc.get("document_type", "unknown")
                    conf = doc.get("confidence", 0)
                    
                    print(f"\n📄 DOCUMENT: {filename}")
                    print(f"   Status: {status}")
                    print(f"   Type: {doc_type} ({conf:.0%} confidence)")
                    print(f"   Description: {doc.get('description', 'N/A')}")
                    
                    # Print extracted data
                    extracted = doc.get("extracted_data", {})
                    if extracted:
                        print(f"\n   📋 EXTRACTED DATA:")
                        print(json.dumps(extracted, indent=6))
                    
                    # Print identifiers
                    identifiers = doc.get("identifiers", {})
                    if identifiers:
                        print(f"\n   🔑 IDENTIFIERS:")
                        for id_type, values in identifiers.items():
                            if values and isinstance(values, list) and len(values) > 0:
                                print(f"      {id_type}: {values}")
                    
                    # Print visible text (truncated)
                    visible_text = doc.get("visible_text", "")
                    if visible_text:
                        print(f"\n   📝 VISIBLE TEXT (first 2000 chars):")
                        print("-" * 60)
                        print(visible_text[:2000])
                        if len(visible_text) > 2000:
                            print(f"... [{len(visible_text) - 2000} more chars]")
                        print("-" * 60)
                
                # Print all identifiers summary
                if all_ids:
                    print(f"\n🔍 ALL IDENTIFIERS (combined):")
                    for id_type, values in all_ids.items():
                        if values and isinstance(values, list) and len(values) > 0:
                            print(f"   {id_type}: {values}")
                
                print("=" * 80 + "\n")
                # =====================================================
                
                # =====================================================
                # BUILD OBSERVATION - PUT IDENTIFIERS AT THE TOP!
                # This is critical: the agent truncates observations,
                # so model numbers MUST be at the very beginning.
                # =====================================================
                
                obs = f"Document analysis complete: {len(docs)} document(s) processed.\n\n"
                
                # ⚠️ CRITICAL: Put extracted identifiers FIRST so agent sees them!
                if all_ids:
                    has_model_numbers = all_ids.get("model_numbers", [])
                    has_order_numbers = all_ids.get("order_numbers", [])
                    has_part_numbers = all_ids.get("part_numbers", [])
                    
                    if has_model_numbers or has_order_numbers or has_part_numbers:
                        obs += "=" * 50 + "\n"
                        obs += "⚠️ CRITICAL - EXTRACTED IDENTIFIERS (USE THESE!):\n"
                        obs += "=" * 50 + "\n"
                        
                        if has_model_numbers:
                            obs += f"🔴 MODEL NUMBERS FOUND: {', '.join(str(v) for v in has_model_numbers)}\n"
                            obs += f"   ➡️ USE THESE MODEL NUMBERS for product_catalog_tool searches!\n"
                            obs += f"   ➡️ DO NOT search generic terms - search these exact models!\n"
                        
                        if has_part_numbers:
                            obs += f"🟠 PART NUMBERS FOUND: {', '.join(str(v) for v in has_part_numbers)}\n"
                        
                        if has_order_numbers:
                            obs += f"🟡 ORDER NUMBERS FOUND: {', '.join(str(v) for v in has_order_numbers)}\n"
                            obs += f"   (Order numbers are NOT product model numbers - don't search catalog with these)\n"
                        
                        obs += "=" * 50 + "\n\n"
                
                # Summary of document types
                if doc_types:
                    type_str = ", ".join([f"{count} {dtype}" for dtype, count in doc_types.items()])
                    obs += f"📊 DOCUMENT TYPES: {type_str}\n\n"
                
                # Per-document details
                for doc in docs:
                    status = doc.get("status", "unknown")
                    if status == "success":
                        doc_type = doc.get("document_type", "unknown")
                        conf = doc.get("confidence", 0)
                        desc = doc.get("description", "")[:300]  # Increased from 200
                        filename = doc.get("filename", "unknown")
                        
                        logger.info(f"[TOOL_EXEC] 📄 {filename}: type={doc_type}, conf={conf:.0%}")
                        
                        obs += f"📄 DOCUMENT: {filename}\n"
                        obs += f"  - Type: {doc_type} ({conf:.0%} confidence)\n"
                        obs += f"  - Description: {desc}\n"
                        
                        # Include identifiers for THIS document prominently
                        identifiers = doc.get("identifiers", {})
                        if identifiers:
                            doc_models = identifiers.get("model_numbers", [])
                            doc_parts = identifiers.get("part_numbers", [])
                            if doc_models:
                                obs += f"  - 🔴 MODEL NUMBERS IN THIS DOC: {', '.join(str(v) for v in doc_models)}\n"
                            if doc_parts:
                                obs += f"  - 🟠 PART NUMBERS IN THIS DOC: {', '.join(str(v) for v in doc_parts)}\n"
                        
                        # Include extracted data summary
                        extracted = doc.get("extracted_data", {})
                        if extracted:
                            obs += f"  - Extracted data: {json.dumps(extracted)[:500]}\n"
                        
                        obs += "\n"
                    else:
                        obs += f"📄 DOCUMENT: {doc.get('filename', 'unknown')} - ERROR: {doc.get('error', 'Unknown error')}\n\n"
                
                # Final reminder at the bottom too
                if all_ids and all_ids.get("model_numbers"):
                    obs += "\n⚠️ REMINDER: Model numbers extracted above. Search product_catalog_tool with these EXACT model numbers!\n"
                
                return output, obs
            else:
                return output, f"Document analysis failed: {output.get('message')}"
        
        # -----------------------------------------------------------
        # 8. OCR IMAGE ANALYZER TOOL (Now Intelligent Image Analyzer)
        # -----------------------------------------------------------
        elif action == "ocr_image_analyzer_tool":
            logger.info(f"[TOOL_EXEC] Executing ocr_image_analyzer_tool (Intelligent Image Analyzer)")
            
            action_input = dict(action_input or {})
            
            # Use ticket images if not provided
            if not action_input.get("image_urls"):
                action_input["image_urls"] = ticket_images
            
            logger.info(f"[TOOL_EXEC] Processing {len(action_input.get('image_urls', []))} image(s) with intelligent analyzer")
            
            output = _run_langchain_tool(ocr_image_analyzer_tool, action_input)
            tool_results["ocr_image_analysis"] = output
            
            if output.get("success"):
                count = output.get('successful_count', len(output.get('results', [])))
                image_types = output.get("image_types", {})
                all_identifiers = output.get("all_identifiers", {})
                summary = output.get("summary", "")
                
                # Log analysis results for debugging
                logger.info(f"[TOOL_EXEC] 📄 Image analyzer processed {count} image(s)")
                logger.info(f"[TOOL_EXEC] 📄 Image types: {image_types}")
                
                # Log each image's analysis
                for result in output.get('results', []):
                    idx = result.get('image_index', '?')
                    img_type = result.get('image_type', 'unknown')
                    confidence = result.get('confidence', 0)
                    description = result.get('description', '')[:200]
                    status = result.get('status', 'unknown')
                    logger.info(f"[TOOL_EXEC] 📄 Image {idx} ({status}): type={img_type}, conf={confidence:.0%}")
                    logger.info(f"[TOOL_EXEC] 📄 Image {idx} description: {description}")
                    
                    identifiers = result.get('identifiers', {})
                    if identifiers.get('model_numbers'):
                        logger.info(f"[TOOL_EXEC] 📄 Image {idx} models: {identifiers['model_numbers']}")
                
                # Build detailed observation for the agent
                # ⚠️ CRITICAL: Put identifiers at TOP so agent sees them!
                obs = f"Image analysis complete: {count} image(s) processed.\n"
                
                # ⚠️ CRITICAL: Put extracted identifiers FIRST!
                if all_identifiers:
                    has_model_numbers = all_identifiers.get("model_numbers", [])
                    has_order_numbers = all_identifiers.get("order_numbers", [])
                    has_part_numbers = all_identifiers.get("part_numbers", [])
                    
                    if has_model_numbers or has_order_numbers or has_part_numbers:
                        obs += "\n" + "=" * 50 + "\n"
                        obs += "⚠️ CRITICAL - EXTRACTED IDENTIFIERS (USE THESE!):\n"
                        obs += "=" * 50 + "\n"
                        
                        if has_model_numbers:
                            obs += f"🔴 MODEL NUMBERS FOUND: {', '.join(str(v) for v in has_model_numbers)}\n"
                            obs += f"   ➡️ USE THESE MODEL NUMBERS for product_catalog_tool searches!\n"
                        
                        if has_part_numbers:
                            obs += f"🟠 PART NUMBERS FOUND: {', '.join(str(v) for v in has_part_numbers)}\n"
                        
                        if has_order_numbers:
                            obs += f"🟡 ORDER NUMBERS FOUND: {', '.join(str(v) for v in has_order_numbers)}\n"
                            obs += f"   (Order numbers are NOT product model numbers - don't search catalog with these)\n"
                        
                        obs += "=" * 50 + "\n"
                
                # Add summary
                if summary:
                    obs += f"\n📊 SUMMARY: {summary}\n"
                
                # Add per-image analysis
                obs += "\n📷 IMAGE ANALYSIS RESULTS:\n"
                for result in output.get('results', []):
                    idx = result.get('image_index', '?')
                    img_type = result.get('image_type', 'unknown')
                    confidence = result.get('confidence', 0)
                    description = result.get('description', 'No description')
                    
                    obs += f"\n--- IMAGE {idx} ---\n"
                    obs += f"• Type: {img_type.upper()} (confidence: {confidence:.0%})\n"
                    obs += f"• Description: {description}\n"
                    
                    # Add identifiers for THIS image prominently
                    identifiers = result.get('identifiers', {})
                    if identifiers:
                        img_models = identifiers.get("model_numbers", [])
                        img_parts = identifiers.get("part_numbers", [])
                        if img_models:
                            obs += f"• 🔴 MODEL NUMBERS IN THIS IMAGE: {', '.join(str(v) for v in img_models)}\n"
                        if img_parts:
                            obs += f"• 🟠 PART NUMBERS IN THIS IMAGE: {', '.join(str(v) for v in img_parts)}\n"
                    
                    # Add extracted data based on image type
                    extracted = result.get('extracted_data', {})
                    if extracted:
                        obs += "• Extracted Data:\n"
                        for key, value in extracted.items():
                            if value:  # Only show non-empty values
                                if isinstance(value, list):
                                    obs += f"    - {key}: {', '.join(str(v) for v in value)}\n"
                                elif isinstance(value, dict):
                                    obs += f"    - {key}: {json.dumps(value)}\n"
                                else:
                                    obs += f"    - {key}: {value}\n"
                    
                    # Add visible text (truncated)
                    visible_text = result.get('visible_text', '')
                    if visible_text:
                        text_preview = visible_text[:300] + "..." if len(visible_text) > 300 else visible_text
                        obs += f"• Visible Text: {text_preview}\n"
                
                # Final reminder
                if all_identifiers.get('model_numbers'):
                    obs += f"\n⚠️ REMINDER: Model numbers extracted above. Search product_catalog_tool with these EXACT model numbers!\n"
                
                return output, obs
            else:
                return output, f"Image analysis failed: {output.get('error', 'Unknown error')}"
        
        # -----------------------------------------------------------
        # 9. FINISH TOOL
        # -----------------------------------------------------------
        elif action == "finish_tool":
            logger.info(f"[TOOL_EXEC] Executing finish_tool")
            output = _run_langchain_tool(finish_tool, action_input or {})
            obs = f"Finished. {output.get('summary', '')}"
            return output, obs
        
        # -----------------------------------------------------------
        # UNKNOWN TOOL
        # -----------------------------------------------------------
        else:
            logger.error(f"[TOOL_EXEC] Unknown tool: {action}")
            obs = f"Unknown tool: {action}"
            return {"error": obs, "success": False}, obs

    except Exception as e:
        logger.error(f"[TOOL_EXEC] Tool execution failed: {e}", exc_info=True)
        obs = f"Tool execution failed: {str(e)}"
        return {"error": obs, "success": False}, obs


def _populate_legacy_fields(
    gathered_documents: List[Dict],
    gathered_images: List,
    gathered_past_tickets: List[Dict],
    identified_product: Dict = None,
    product_confidence: float = 0.0,
    gemini_answer: str = ""
) -> Dict[str, Any]:
    """
    Populate legacy RAG result fields for compatibility with existing nodes.
    """
    
    # Normalize and validate inputs
    product_details = identified_product or {}
    relevant_documents = _normalize_documents(gathered_documents)
    relevant_images = _normalize_images(gathered_images)
    past_tickets = _normalize_tickets(gathered_past_tickets)
    
    # Deduplicate documents by title
    seen_titles = set()
    unique_docs = []
    for doc in relevant_documents:
        title = doc.get("title", "").lower()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_docs.append(doc)
        elif not title:  # Allow docs without titles
            unique_docs.append(doc)
    relevant_documents = unique_docs
    
    # Convert to RetrievalHit format for legacy nodes
    text_retrieval_results = []
    for i, doc in enumerate(relevant_documents):
        text_retrieval_results.append({
            "id": doc.get("id", f"doc_{i}"),
            "score": doc.get("relevance_score", 0.8),
            "metadata": {
                "title": doc.get("title", "Unknown"),
                "source": "gemini_file_search"
            },
            "content": doc.get("content_preview", doc.get("title", ""))
        })
    
    # Convert images to RetrievalHit format
    image_retrieval_results = []
    for i, img_url in enumerate(relevant_images):
        if img_url:
            image_retrieval_results.append({
                "id": f"img_{i}",
                "score": 0.9,
                "metadata": {
                    "image_url": img_url,
                    "source": "react_vision"
                },
                "content": f"Product image {i+1}"
            })
    
    # Convert past tickets to RetrievalHit format
    past_ticket_results = []
    for i, ticket in enumerate(past_tickets):
        similarity = ticket.get("similarity_score", 0)
        if isinstance(similarity, (int, float)) and similarity > 1:
            similarity = similarity / 100.0
        
        past_ticket_results.append({
            "id": f"ticket_{ticket.get('ticket_id', i)}",
            "score": similarity,
            "metadata": {
                "ticket_id": ticket.get("ticket_id"),
                "subject": ticket.get("subject"),
                "resolution_type": ticket.get("resolution_type"),
                "source": "past_tickets"
            },
            "content": ticket.get("resolution_summary", "")
        })
    
    # Build multimodal_context string
    context_sections = []
    
    if gemini_answer:
        context_sections.append("### 🎯 DIRECT ANSWER FROM DOCUMENTATION")
        context_sections.append(str(gemini_answer)[:1000])
        context_sections.append("")
    
    if text_retrieval_results:
        context_sections.append("### PRODUCT DOCUMENTATION")
        for i, hit in enumerate(text_retrieval_results[:10], 1):
            title = hit.get("metadata", {}).get("title", f"Document {i}")
            content = hit.get("content", "")[:500]
            score = hit.get("score", 0.0)
            context_sections.append(f"{i}. **{title}** (score: {score:.2f})\n{content}\n")
    
    if image_retrieval_results and identified_product:
        context_sections.append("\n### PRODUCT MATCHES (VISUAL)")
        model = identified_product.get("model", "Unknown")
        name = identified_product.get("name", "Product")
        category = identified_product.get("category", "Unknown")
        context_sections.append(f"Identified Product: {name} (Model: {model}, Category: {category})")
        context_sections.append(f"Confidence: {product_confidence:.2%}")

    if past_ticket_results:
        context_sections.append("\n### SIMILAR PAST TICKETS")
        for i, hit in enumerate(past_ticket_results[:3], 1):
            meta = hit.get("metadata", {})
            ticket_id = meta.get("ticket_id", "Unknown")
            resolution_type = meta.get("resolution_type", "N/A")
            content = hit.get("content", "")[:300]
            score = hit.get("score", 0.0)
            context_sections.append(
                f"{i}. Ticket #{ticket_id} ({resolution_type}) - Similarity: {score:.2f}\n{content}\n"
            )
    
    multimodal_context = "\n".join(context_sections) if context_sections else "No relevant context found."
    if not multimodal_context or len(multimodal_context.strip()) < 50:
        logger.warning("Multimodal context is empty or too short!")
        multimodal_context = "No context gathered. Agent did not retrieve sufficient information."
    
    # Build source fields
    source_documents = []
    for i, doc in enumerate(relevant_documents[:10]):
        source_documents.append({
            "rank": i + 1,
            "title": doc.get("title", "Unknown"),
            "content_preview": str(doc.get("content_preview", ""))[:500],
            "relevance_score": doc.get("relevance_score", 0),
            "source_type": "gemini_file_search",
            "uri": doc.get("uri", "")
        })
    
    source_products = []
    if identified_product:
        source_products.append({
            "rank": 1,
            "model_no": identified_product.get("model", "Unknown"),
            "product_title": identified_product.get("name", "Unknown"),
            "category": identified_product.get("category", "Unknown"),
            "similarity_score": int(product_confidence * 100),
            "source_type": "react_agent"
        })
    
    source_tickets = []
    for i, ticket in enumerate(past_tickets[:5]):
        source_tickets.append({
            "rank": i + 1,
            "ticket_id": ticket.get("ticket_id"),
            "subject": ticket.get("subject"),
            "resolution_type": ticket.get("resolution_type"),
            "resolution_summary": str(ticket.get("resolution_summary", ""))[:200],
            "similarity_score": ticket.get("similarity_score", 0),
            "source_type": "past_tickets"
        })
    
    has_docs = len(text_retrieval_results) > 0
    has_images = len(image_retrieval_results) > 0
    has_product = identified_product is not None
    enough_info = has_docs or has_images or has_product
    
    logger.info(f"[LEGACY_FIELDS] Populated: docs={len(text_retrieval_results)}, "
                f"images={len(image_retrieval_results)}, tickets={len(past_ticket_results)}, "
                f"context_len={len(multimodal_context)}, enough_info={enough_info}")
    
    return {
        "text_retrieval_results": text_retrieval_results,
        "image_retrieval_results": image_retrieval_results,
        "past_ticket_results": past_ticket_results,
        "multimodal_context": multimodal_context,
        "source_documents": source_documents,
        "source_products": source_products,
        "source_tickets": source_tickets,
        "gemini_answer": gemini_answer,
        "enough_information": enough_info,
        "product_match_confidence": product_confidence,
        "overall_confidence": product_confidence,
        # Set hallucination_risk based on confidence (low confidence = higher risk)
        # This replaces the removed hallucination_guard node
        "hallucination_risk": max(0.0, 1.0 - product_confidence) if product_confidence > 0 else 0.3,
        "ran_vision": True,
        "ran_text_rag": True,
        "ran_past_tickets": True
    }


def _normalize_documents(docs: List[Any]) -> List[Dict[str, Any]]:
    if not docs: return []
    normalized = []
    for doc in docs:
        if isinstance(doc, dict):
            normalized.append(doc)
        elif isinstance(doc, str):
            normalized.append({"id": doc, "title": doc, "content_preview": ""})
        else:
            normalized.append({"id": str(doc), "title": str(doc), "content_preview": ""})
    return normalized


def _normalize_images(images: List[Any]) -> List[str]:
    if not images: return []
    normalized = []
    for img in images:
        if isinstance(img, dict):
            url = img.get("url") or img.get("image_url") or img.get("src") or ""
            if url:
                normalized.append(url)
        elif isinstance(img, str):
            normalized.append(img)
    return normalized


def _normalize_tickets(tickets: List[Any]) -> List[Dict[str, Any]]:
    if not tickets: return []
    normalized = []
    for ticket in tickets:
        if isinstance(ticket, dict):
            normalized.append(ticket)
        elif isinstance(ticket, str):
            normalized.append({"ticket_id": ticket, "subject": "Unknown", "resolution_summary": ""})
        else:
            normalized.append({"ticket_id": str(ticket), "subject": "Unknown", "resolution_summary": ""})
    return normalized