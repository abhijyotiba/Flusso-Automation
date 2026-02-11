"""
State Model for LangGraph Workflow
Defines the complete state structure for ticket processing
Enhanced with ReACT agent fields for intelligent tool orchestration
"""

from typing import TypedDict, List, Dict, Any, Optional


class RetrievalHit(TypedDict):
    """Single retrieval result from vector database or file search."""
    id: str
    score: float
    metadata: Dict[str, Any]
    content: str  # Text chunk, product info, or ticket summary


class ReACTIteration(TypedDict):
    """Single ReACT reasoning iteration tracking"""
    iteration: int
    thought: str                          # Agent's reasoning
    action: str                           # Tool name called
    action_input: Dict[str, Any]         # Tool parameters
    observation: str                      # Tool output summary
    tool_output: Dict[str, Any]          # Full tool result
    timestamp: float                      # When this happened
    duration: float                       # How long tool took


class TicketState(TypedDict, total=False):
    """
    Complete state object passed between all LangGraph nodes.
    Each node reads from and updates this state.
    """

    # ==========================================
    # RAW TICKET INFO
    # ==========================================
    ticket_id: str
    ticket_subject: str
    ticket_text: str  # Includes description + extracted attachment content
    ticket_images: List[str]
    requester_email: str
    requester_name: str
    ticket_type: Optional[str]
    priority: Optional[str]
    tags: List[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    ticket_conversation: List[Dict[str, Any]]  # OPTIONAL but recommended
    ticket_attachments: List[Dict[str, Any]]  # Original attachment metadata
    
    # ==========================================
    # PLANNING MODULE (Phase 1 Enhancement)
    # ==========================================
    execution_plan: Optional[Dict[str, Any]]      # Full plan from planner
    plan_steps: List[Dict[str, Any]]              # Ordered steps to execute
    current_plan_step: int                         # Current step index (0-based)
    ticket_complexity: Optional[str]               # "simple" | "moderate" | "complex"
    planning_confidence: float                     # Planner's confidence (0.0-1.0)
    
    # Policy context
    applicable_policy_type: Optional[str]          # "warranty" | "return" | "missing_parts" | etc.
    policy_requirements: List[str]                 # What policy requires (e.g., "proof of purchase")
    policy_context: Optional[str]                  # Full policy section text
    can_proceed_per_policy: bool                   # Whether we have what policy needs
    missing_for_policy: List[str]                  # What's missing per policy
    
    # Analysis results from planner
    customer_need_analysis: Optional[str]          # What the customer wants
    help_type: Optional[str]                       # warranty|return|parts|installation|etc.
    mentioned_product_model: Optional[str]         # Model number if found in ticket
    
    product_match_confidence: float
    product_match_reasoning: str
    
    # ==========================================
    # TICKET FACTS (from ticket_extractor node)
    # Mutable record of what's known about the ticket
    # ==========================================
    ticket_facts: Optional[Dict[str, Any]]
    # Structure:
    # {
    #   --- TIER 1: DETERMINISTIC (set by ticket_extractor) ---
    #   "has_address": bool,              # Address keywords detected
    #   "has_receipt": bool,              # Receipt/invoice keywords detected
    #   "has_po": bool,                   # PO number keywords detected
    #   "has_video": bool,                # Video attachment present
    #   "has_photos": bool,               # Image attachments present
    #   "has_document_attachments": bool, # Non-image attachments present
    #   "customer_name": str,             # From Freshdesk
    #   "requester_email": str,           # From Freshdesk
    #   "extracted_address": str | None,  # Regex-extracted address
    #   "address_confidence": float,      # 0.0-1.0
    #   "raw_product_codes": List[{       # Product codes found in text
    #       "full_sku": str,              # e.g., "TRM.TVH.0211BB"
    #       "model": str,                 # e.g., "TRM.TVH.0211"
    #       "finish_code": str | None,    # e.g., "BB"
    #       "finish_name": str | None     # e.g., "Brushed Bronze PVD"
    #   }],
    #   "raw_part_numbers": List[str],    # Part numbers found
    #   "raw_finish_mentions": List[str], # Finish names mentioned in text
    #
    #   --- TIER 2: LLM-VERIFIED (set by planner) ---
    #   "planner_verified": bool,
    #   "planner_verified_models": List[str],
    #   "planner_verified_finishes": List[str],
    #   "planner_corrections": Dict,
    #
    #   --- TIER 3: TOOL-CONFIRMED (set by react_agent) ---
    #   "confirmed_model": str | None,
    #   "confirmed_model_source": str | None,  # "catalog", "ocr", "vision", "text"
    #   "confirmed_model_confidence": float,
    #   "confirmed_finish": str | None,
    #   "confirmed_finish_name": str | None,
    #   "confirmed_parts": List[str],
    #
    #   --- METADATA ---
    #   "extraction_version": str,
    #   "extracted_at": float,
    #   "last_updated_at": float,
    #   "last_updated_by": str,
    #   "update_history": List[Dict]
    # }
    
    # ==========================================
    # ATTACHMENT INFO
    # ==========================================
    attachment_summary: List[Dict[str, Any]]  # List of processed attachments with metadata

    # ==========================================
    # REACT AGENT FIELDS (NEW)
    # ==========================================
    react_iterations: List[ReACTIteration]      # Full reasoning chain
    react_total_iterations: int                  # Count of iterations
    react_status: str                            # "pending" | "running" | "finished" | "max_iterations"
    react_final_reasoning: str                   # Why agent stopped
    
    # Product Identification (from ReACT)
    identified_product: Optional[Dict[str, Any]]  # {model, name, category, confidence}
    product_identification_method: Optional[str]  # "text" | "vision" | "attachment" | "metadata"
    product_confidence: float                     # 0.0 - 1.0
    
    # Gathered Resources (from ReACT)
    gathered_documents: List[Dict[str, Any]]      # Relevant docs with links
    gathered_images: List[str]                    # Product image URLs
    gathered_past_tickets: List[Dict[str, Any]]   # Similar resolved tickets
    
    # Attachment Analysis (from ReACT)
    attachment_analysis: Dict[str, Any]           # Extracted model numbers, entities

    # ==========================================
    # RAG EXECUTION FLAGS (REQUIRED for sequential mode)
    # ==========================================
    ran_vision: bool
    ran_text_rag: bool
    ran_past_tickets: bool

    # ==========================================
    # CLASSIFICATION / ROUTING
    # ==========================================
    ticket_category: Optional[str]
    has_text: bool
    has_image: bool
    
    # ==========================================
    # SKIP LOGIC (for PO, auto-reply, spam)
    # ==========================================
    should_skip: bool  # True if ticket should skip full workflow
    skip_reason: Optional[str]  # Why the ticket was skipped
    skip_private_note: Optional[str]  # Private note for skipped tickets
    skip_workflow_applied: bool  # Flag set by skip_handler node
    category_requires_vision: bool  # Whether category needs vision pipeline
    category_requires_text_rag: bool  # Whether category needs text RAG
    
    # Fields set by skip_handler
    suggested_tags: List[str]  # Tags suggested for skipped tickets
    private_note: Optional[str]  # Private note content
    resolution_decision: Optional[str]  # skip_workflow, resolved, etc.

    # ==========================================
    # CUSTOMER PROFILE / RULES
    # ==========================================
    customer_type: Optional[str]
    customer_metadata: Dict[str, Any]
    customer_rules: Dict[str, Any]  # Dealer/End Customer business rules from customer_rules.py

    # ==========================================
    # RAG RESULTS
    # ==========================================
    text_retrieval_results: List[RetrievalHit]
    image_retrieval_results: List[RetrievalHit]
    past_ticket_results: List[RetrievalHit]
    multimodal_context: str
    
    # ==========================================
    # SOURCE CITATIONS (for enhanced response)
    # ==========================================
    gemini_answer: Optional[str]  # Raw answer from Gemini file search
    source_documents: List[Dict[str, Any]]  # Structured docs from Gemini grounding
    source_products: List[Dict[str, Any]]   # Products from vision_search_tool ONLY (Visual Matches section)
    source_tickets: List[Dict[str, Any]]    # Structured tickets from Past Tickets
    
    # ==========================================
    # VISION MATCH QUALITY (new fields)
    # ==========================================
    vision_match_quality: Optional[str]  # "HIGH", "LOW", "NO_MATCH", "CATEGORY_MISMATCH"
    vision_relevance_reason: Optional[str]  # Explanation for the quality assessment
    vision_matched_category: Optional[str]  # What category the vision matched (e.g., "Sink Faucets")
    vision_expected_category: Optional[str]  # What the customer is asking about (e.g., "Shower Hinges")

    # ==========================================
    # EVIDENCE RESOLUTION (conflict handling)
    # ==========================================
    evidence_analysis: Optional[Dict[str, Any]]  # Full analysis from evidence resolver
    needs_more_info: bool                         # True if conflicting evidence needs customer clarification
    info_request_response: Optional[str]          # Customer-facing message asking for more info
    evidence_decision: Optional[str]              # "ACCEPT_OCR", "ACCEPT_VISION", "REQUIRES_INFO", etc.
    missing_requirements: List[str]               # Missing info for return/replacement (PO, photo, address)
    
    # ==========================================
    # CONSTRAINT VALIDATION (from constraint_validator)
    # ==========================================
    constraint_result: Optional[Dict[str, Any]]   # Full result from constraint_validator
    # Structure:
    # {
    #   "original_category": str,
    #   "resolved_category": str,
    #   "missing_fields": List[str],         # Fields that are required but missing
    #   "required_asks": List[str],          # Customer-friendly ask messages
    #   "present_fields": List[str],         # Fields that ARE present
    #   "must_not_ask": List[str],           # What NOT to ask for (already provided)
    #   "applicable_policies": List[str],    # Policy keys that apply
    #   "policy_citations": List[Dict],      # {policy_id, name, citation}
    #   "required_citations": List[str],     # Citation text that MUST appear
    #   "conditional_fields": Dict[str,str], # Fields that might be needed
    #   "can_proceed": bool,                 # Whether we have minimum info
    #   "blocking_missing": List[str],       # Critical missing fields
    #   "validation_notes": List[str],       # Notes from validation
    # }
    constraints_prompt_section: Optional[str]     # Formatted constraints for LLM prompt

    # ==========================================
    # PRODUCT / DECISION METRICS
    # ==========================================
    detected_product_id: Optional[str]   # OPTIONAL but recommended
    product_match_confidence: float
    hallucination_risk: float
    enough_information: bool
    overall_confidence: float  # Combined confidence score (0-100%)

    # ==========================================
    # LLM OUTPUTS
    # ==========================================
    clarification_message: Optional[str]
    draft_response: Optional[str]

    # ==========================================
    # FINAL OUTCOME
    # ==========================================
    final_response_public: Optional[str]
    final_private_note: Optional[str]
    resolution_status: Optional[str]
    extra_tags: List[str]

    # ==========================================
    # WORKFLOW ERROR TRACKING
    # ==========================================
    workflow_error: Optional[str]              # Error message if workflow failed
    workflow_error_type: Optional[str]         # "api_error" | "timeout" | "rate_limit" | "internal"
    workflow_error_node: Optional[str]         # Which node failed
    is_system_error: bool                       # True = system failure, False = legitimate need-more-info
    
    # ==========================================
    # AUDIT TRAIL
    # ==========================================
    audit_events: List[Dict[str, Any]]
