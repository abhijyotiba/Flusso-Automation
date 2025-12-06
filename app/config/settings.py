"""
Application Settings using Pydantic
Loads and validates all environment variables
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application configuration from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ==========================================
    # FRESHDESK
    # ==========================================
    freshdesk_domain: str
    freshdesk_api_key: str
    
    # ==========================================
    # PINECONE
    # ==========================================
    pinecone_api_key: str
    pinecone_env: str = "us-east-1"
    pinecone_image_index: str
    pinecone_tickets_index: str
    
    # ==========================================
    # GEMINI
    # ==========================================
    gemini_api_key: str
    gemini_file_search_store_id: str
    
    # ==========================================
    # OPENAI (for embeddings - optional)
    # ==========================================
    openai_api_key: Optional[str] = None
    
    # ==========================================
    # APPLICATION
    # ==========================================
    environment: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # ==========================================
    # RAG SETTINGS
    # ==========================================
    text_retrieval_top_k: int = 10
    image_retrieval_top_k: int = 4
    past_ticket_top_k: int = 3
    
    # ==========================================
    # DECISION THRESHOLDS
    # More lenient since AI is a companion for human agents,
    # not directly responding to customers
    # ==========================================
    hallucination_risk_threshold: float = 0.7  # Allow up to 70% risk (agent reviews anyway)
    product_confidence_threshold: float = 0.4  # Show results even with 40%+ confidence
    
    # ==========================================
    # LLM SETTINGS
    # ==========================================
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 8192  # Increased for complete structured responses
    
    # ==========================================
    # CLIP SETTINGS (for image embeddings - 512 dimensions)
    # ==========================================
    clip_model: str = "ViT-B-32"  # 512 dimensions for image index
    clip_pretrained: str = "openai"
    gpu_enabled: bool = False
    
    # ==========================================
    # VISION PIPELINE SETTINGS
    # ==========================================
    vision_min_similarity_threshold: float = 0.75  # Minimum score to consider a match valid
    vision_category_validation: bool = True  # Enable LLM category validation
    
    # ==========================================
    # VERTEX AI SETTINGS (production multimodal embeddings)
    # Set USE_VERTEX_AI_EMBEDDINGS=true to use Vertex AI instead of CLIP
    # ==========================================
    vertex_ai_project: Optional[str] = None
    vertex_ai_location: str = "us-central1"
    use_vertex_ai_embeddings: bool = False  # Toggle for production
    vertex_ai_embedding_dimension: int = 512  # Match CLIP/Pinecone index
    
    def validate_all(self) -> None:
        """Validate critical settings with comprehensive checks"""
        errors = []
        warnings = []
        
        # Required API credentials
        if not self.freshdesk_domain:
            errors.append("FRESHDESK_DOMAIN is required")
        elif not self.freshdesk_domain.startswith("https://"):
            errors.append("FRESHDESK_DOMAIN must start with 'https://'")
            
        if not self.freshdesk_api_key:
            errors.append("FRESHDESK_API_KEY is required")
        if not self.pinecone_api_key:
            errors.append("PINECONE_API_KEY is required")
        if not self.gemini_api_key:
            errors.append("GEMINI_API_KEY is required")
        if not self.gemini_file_search_store_id:
            errors.append("GEMINI_FILE_SEARCH_STORE_ID is required")
        
        # Validate threshold ranges (0.0 to 1.0)
        if not (0.0 <= self.hallucination_risk_threshold <= 1.0):
            errors.append(f"hallucination_risk_threshold must be between 0.0 and 1.0, got {self.hallucination_risk_threshold}")
        if not (0.0 <= self.product_confidence_threshold <= 1.0):
            errors.append(f"product_confidence_threshold must be between 0.0 and 1.0, got {self.product_confidence_threshold}")
        if not (0.0 <= self.vision_min_similarity_threshold <= 1.0):
            errors.append(f"vision_min_similarity_threshold must be between 0.0 and 1.0, got {self.vision_min_similarity_threshold}")
        if not (0.0 <= self.llm_temperature <= 2.0):
            errors.append(f"llm_temperature must be between 0.0 and 2.0, got {self.llm_temperature}")
        
        # Validate retrieval counts
        if self.text_retrieval_top_k < 1 or self.text_retrieval_top_k > 100:
            warnings.append(f"text_retrieval_top_k={self.text_retrieval_top_k} seems unusual (expected 1-100)")
        if self.image_retrieval_top_k < 1 or self.image_retrieval_top_k > 50:
            warnings.append(f"image_retrieval_top_k={self.image_retrieval_top_k} seems unusual (expected 1-50)")
        
        # Vertex AI validation
        if self.use_vertex_ai_embeddings:
            if not self.vertex_ai_project:
                errors.append("VERTEX_AI_PROJECT is required when USE_VERTEX_AI_EMBEDDINGS=true")
            if self.vertex_ai_embedding_dimension not in [128, 256, 512, 768, 1024]:
                warnings.append(f"vertex_ai_embedding_dimension={self.vertex_ai_embedding_dimension} is non-standard")
        
        # Log warnings
        if warnings:
            import logging
            logger = logging.getLogger(__name__)
            for w in warnings:
                logger.warning(f"Configuration warning: {w}")
            
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))


# Global settings instance
settings = Settings()
