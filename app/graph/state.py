"""
State Model for LangGraph Workflow
Defines the complete state structure for ticket processing
"""

from typing import TypedDict, List, Dict, Any, Optional


class RetrievalHit(TypedDict):
    """Single retrieval result from vector database or file search."""
    id: str
    score: float
    metadata: Dict[str, Any]
    content: str  # Text chunk, product info, or ticket summary


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
    
    # ==========================================
    # ATTACHMENT INFO
    # ==========================================
    attachment_summary: List[Dict[str, Any]]  # List of processed attachments with metadata

    # ==========================================
    # RAG EXECUTION FLAGS (REQUIRED!)
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
    vip_rules: Dict[str, Any]

    # ==========================================
    # RAG RESULTS
    # ==========================================
    text_retrieval_results: List[RetrievalHit]
    image_retrieval_results: List[RetrievalHit]
    past_ticket_results: List[RetrievalHit]
    multimodal_context: str
    
    # ==========================================
    # VISION MATCH QUALITY (new fields)
    # ==========================================
    vision_match_quality: Optional[str]  # "HIGH", "LOW", "NO_MATCH", "CATEGORY_MISMATCH"
    vision_relevance_reason: Optional[str]  # Explanation for the quality assessment
    vision_matched_category: Optional[str]  # What category the vision matched (e.g., "Sink Faucets")
    vision_expected_category: Optional[str]  # What the customer is asking about (e.g., "Shower Hinges")

    # ==========================================
    # PRODUCT / DECISION METRICS
    # ==========================================
    detected_product_id: Optional[str]   # OPTIONAL but recommended
    product_match_confidence: float
    hallucination_risk: float
    enough_information: bool
    vip_compliant: bool
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
    # AUDIT TRAIL
    # ==========================================
    audit_events: List[Dict[str, Any]]
