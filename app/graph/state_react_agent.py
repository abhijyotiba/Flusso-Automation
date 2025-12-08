"""
State Model for LangGraph Workflow with ReACT Agent
Enhanced with ReACT reasoning fields
"""

from typing import TypedDict, List, Dict, Any, Optional


class RetrievalHit(TypedDict):
    """Single retrieval result from vector database or file search."""
    id: str
    score: float
    metadata: Dict[str, Any]
    content: str


class ReACTIteration(TypedDict):
    """Single ReACT reasoning iteration"""
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
    Enhanced with ReACT agent fields.
    """

    # ==========================================
    # RAW TICKET INFO
    # ==========================================
    ticket_id: str
    ticket_subject: str
    ticket_text: str
    ticket_images: List[str]
    requester_email: str
    requester_name: str
    ticket_type: Optional[str]
    priority: Optional[str]
    tags: List[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    
    # ==========================================
    # ATTACHMENT INFO
    # ==========================================
    attachment_summary: List[Dict[str, Any]]
    
    # ==========================================
    # REACT AGENT FIELDS (NEW)
    # ==========================================
    react_iterations: List[ReACTIteration]      # Full reasoning chain
    react_total_iterations: int                  # Count
    react_status: str                            # "running" | "finished" | "max_iterations"
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
    # CLASSIFICATION / ROUTING
    # ==========================================
    ticket_category: Optional[str]
    has_text: bool
    has_image: bool
    
    # ==========================================
    # SKIP LOGIC
    # ==========================================
    should_skip: bool
    skip_reason: Optional[str]
    skip_private_note: Optional[str]
    skip_workflow_applied: bool
    
    # ==========================================
    # CUSTOMER PROFILE / RULES
    # ==========================================
    customer_type: Optional[str]
    customer_metadata: Dict[str, Any]
    vip_rules: Dict[str, Any]
    
    # ==========================================
    # LEGACY RAG RESULTS (Still populated for compatibility)
    # ==========================================
    text_retrieval_results: List[RetrievalHit]
    image_retrieval_results: List[RetrievalHit]
    past_ticket_results: List[RetrievalHit]
    multimodal_context: str
    
    # ==========================================
    # SOURCE CITATIONS
    # ==========================================
    gemini_answer: Optional[str]
    source_documents: List[Dict[str, Any]]
    source_products: List[Dict[str, Any]]
    source_tickets: List[Dict[str, Any]]
    
    # ==========================================
    # VISION MATCH QUALITY (Legacy, still used by draft_response)
    # ==========================================
    vision_match_quality: Optional[str]
    vision_relevance_reason: Optional[str]
    vision_matched_category: Optional[str]
    vision_expected_category: Optional[str]
    
    # ==========================================
    # DECISION METRICS
    # ==========================================
    detected_product_id: Optional[str]
    product_match_confidence: float
    hallucination_risk: float
    enough_information: bool
    vip_compliant: bool
    overall_confidence: float
    
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
    suggested_tags: List[str]
    private_note: Optional[str]
    resolution_decision: Optional[str]
    
    # ==========================================
    # AUDIT TRAIL
    # ==========================================
    audit_events: List[Dict[str, Any]]
