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
    image_retrieval_top_k: int = 5
    past_ticket_top_k: int = 5
    
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
    llm_model: str = "gemini-2.0-flash-exp"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048
    
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
        """Validate critical settings"""
        errors = []
        
        if not self.freshdesk_domain:
            errors.append("FRESHDESK_DOMAIN is required")
        if not self.freshdesk_api_key:
            errors.append("FRESHDESK_API_KEY is required")
        if not self.pinecone_api_key:
            errors.append("PINECONE_API_KEY is required")
        if not self.gemini_api_key:
            errors.append("GEMINI_API_KEY is required")
        if not self.gemini_file_search_store_id:
            errors.append("GEMINI_FILE_SEARCH_STORE_ID is required")
            
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))


# Global settings instance
settings = Settings()
