"""
Multi-Provider Image & Text Embeddings

Image Embeddings (512 dimensions):
- CLIP (local): Default for development, runs on CPU/GPU
- Vertex AI (cloud): Production option, uses Google Cloud

Text Embeddings (768 dimensions):
- Gemini: For text/tickets index

Toggle between CLIP and Vertex AI using USE_VERTEX_AI_EMBEDDINGS env var.
"""

import logging
import torch
import open_clip
from PIL import Image
from pathlib import Path
from typing import Dict, List, Union, Optional, Protocol
import numpy as np
from io import BytesIO
import requests
from google import genai
from abc import ABC, abstractmethod

from app.config.settings import settings

logger = logging.getLogger(__name__)


# =====================================================
# EMBEDDER INTERFACE (Abstract Base)
# =====================================================

class ImageEmbedderInterface(ABC):
    """Interface for image embedding providers"""
    
    @abstractmethod
    def embed_image_from_path(self, image_path: Union[str, Path]) -> np.ndarray:
        """Generate embedding for image from local path"""
        pass
    
    @abstractmethod
    def embed_image_from_url(self, image_url: str) -> np.ndarray:
        """Generate embedding for image from URL"""
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate text embedding in same space as images"""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimension of embedding vectors"""
        pass


# =====================================================
# CLIP EMBEDDER (Local - Development)
# =====================================================

class CLIPEmbedder(ImageEmbedderInterface):
    """
    CLIP-based image embedder with optional GPU support.
    Generates normalized embeddings for product images (512 dimensions).
    Default for local development - no cloud credentials needed.
    """
    
    def __init__(self):
        """Initialize CLIP model"""
        self.device = self._setup_device()
        self.model, self.preprocess = self._load_model()
        logger.info(f"CLIP Embedder initialized on device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Configure device (GPU/CPU) for inference"""
        if settings.gpu_enabled and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            if settings.gpu_enabled:
                logger.warning("GPU requested but not available. Using CPU.")
            else:
                logger.info("Using CPU for embeddings")
        
        return device
    
    def _load_model(self):
        """Load CLIP model and preprocessing"""
        logger.info(f"Loading CLIP model: {settings.clip_model}")
        
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                settings.clip_model,
                pretrained=settings.clip_pretrained,
                device=self.device
            )
            
            model.eval()
            
            # Test embedding dimension
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                embedding_dim = model.encode_image(dummy_input).shape[-1]
            
            logger.info(f"CLIP model loaded. Embedding dimension: {embedding_dim}")
            
            return model, preprocess
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def embed_image_from_path(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Generate embedding for image from local path
        
        Args:
            image_path: Path to image file
            
        Returns:
            Normalized embedding vector (numpy array)
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return self._embed_pil_image(image)
            
        except Exception as e:
            logger.error(f"Failed to embed image from path {image_path}: {e}")
            raise
    
    def embed_image_from_url(self, image_url: str) -> np.ndarray:
        """
        Generate embedding for image from URL
        
        Args:
            image_url: HTTP(S) URL to image
            
        Returns:
            Normalized embedding vector (numpy array)
        """
        try:
            # Download image
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Load image from bytes
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return self._embed_pil_image(image)
            
        except Exception as e:
            logger.error(f"Failed to embed image from URL {image_url}: {e}")
            raise
    
    def _embed_pil_image(self, image: Image.Image) -> np.ndarray:
        """
        Generate embedding for PIL Image
        
        Args:
            image: PIL Image object
            
        Returns:
            Normalized embedding vector
        """
        try:
            # Preprocess and convert to tensor
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
                # Normalize
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            # Convert to numpy
            embedding_np = embedding.cpu().numpy().flatten()
            
            return embedding_np
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embedding vectors"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            embedding = self.model.encode_image(dummy_input)
            return embedding.shape[-1]
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate text embedding using CLIP (512 dimensions).
        Text is embedded in the same space as images.
        
        Args:
            text: Text to embed
            
        Returns:
            Normalized embedding vector (numpy array)
        """
        tokenizer = open_clip.get_tokenizer(settings.clip_model)
        
        try:
            with torch.no_grad():
                text_tokens = tokenizer([text]).to(self.device)
                text_embedding = self.model.encode_text(text_tokens)
                text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                return text_embedding.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Failed to embed text with CLIP: {e}")
            raise


# =====================================================
# VERTEX AI EMBEDDER (Cloud - Production)
# =====================================================

class VertexAIEmbedder(ImageEmbedderInterface):
    """
    Vertex AI Multimodal Embeddings for production.
    Uses Google Cloud's multimodalembedding@001 model.
    
    Key benefits over CLIP:
    - Text and images in the SAME semantic space
    - No local GPU needed
    - Enterprise-grade reliability
    - Configurable dimensions (128, 256, 512, 1408)
    
    Requires:
    - GCP Project with Vertex AI API enabled
    - Application Default Credentials (gcloud auth application-default login)
    """
    
    _initialized: bool = False
    _model = None
    
    def __init__(self):
        """Initialize Vertex AI multimodal embedding model"""
        self._init_vertex_ai()
        logger.info(f"Vertex AI Embedder initialized (project: {settings.vertex_ai_project}, dim: {settings.vertex_ai_embedding_dimension})")
    
    def _init_vertex_ai(self):
        """Initialize Vertex AI SDK and load model"""
        if VertexAIEmbedder._initialized:
            return
        
        try:
            import vertexai
            from vertexai.vision_models import MultiModalEmbeddingModel
            
            # Initialize Vertex AI with project and location
            if not settings.vertex_ai_project:
                raise ValueError("VERTEX_AI_PROJECT not configured")
            
            vertexai.init(
                project=settings.vertex_ai_project,
                location=settings.vertex_ai_location
            )
            
            # Load the multimodal embedding model
            VertexAIEmbedder._model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
            VertexAIEmbedder._initialized = True
            
            logger.info("Vertex AI multimodal embedding model loaded")
            
        except ImportError as e:
            logger.error("google-cloud-aiplatform not installed. Run: pip install google-cloud-aiplatform")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
    
    def embed_image_from_path(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Generate embedding for image from local path using Vertex AI.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Normalized embedding vector (numpy array)
        """
        try:
            from vertexai.vision_models import Image as VertexImage
            
            # Load image using Vertex AI's Image class
            image = VertexImage.load_from_file(str(image_path))
            
            # Get embeddings with configured dimension
            embeddings = VertexAIEmbedder._model.get_embeddings(
                image=image,
                dimension=settings.vertex_ai_embedding_dimension
            )
            
            return np.array(embeddings.image_embedding, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Vertex AI: Failed to embed image from path {image_path}: {e}")
            raise
    
    def embed_image_from_url(self, image_url: str) -> np.ndarray:
        """
        Generate embedding for image from URL using Vertex AI.
        Downloads image first, then embeds.
        
        Args:
            image_url: HTTP(S) URL to image
            
        Returns:
            Normalized embedding vector (numpy array)
        """
        try:
            from vertexai.vision_models import Image as VertexImage
            
            # Download image to bytes
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Save to temp file (Vertex AI needs file path or GCS URI)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            
            try:
                image = VertexImage.load_from_file(tmp_path)
                embeddings = VertexAIEmbedder._model.get_embeddings(
                    image=image,
                    dimension=settings.vertex_ai_embedding_dimension
                )
                return np.array(embeddings.image_embedding, dtype=np.float32)
            finally:
                # Cleanup temp file
                import os
                os.unlink(tmp_path)
            
        except Exception as e:
            logger.error(f"Vertex AI: Failed to embed image from URL {image_url}: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate text embedding using Vertex AI (same space as images!).
        This is a KEY advantage - text and images are directly comparable.
        
        Args:
            text: Text to embed (max 32 tokens)
            
        Returns:
            Embedding vector (numpy array)
        """
        try:
            embeddings = VertexAIEmbedder._model.get_embeddings(
                contextual_text=text,
                dimension=settings.vertex_ai_embedding_dimension
            )
            
            return np.array(embeddings.text_embedding, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Vertex AI: Failed to embed text: {e}")
            raise
    
    def get_embedding_dim(self) -> int:
        """Get the configured embedding dimension"""
        return settings.vertex_ai_embedding_dimension


# =====================================================
# GLOBAL EMBEDDER INSTANCES & FACTORY
# =====================================================

_clip_embedder: Dict[str, CLIPEmbedder] = {}
_vertex_embedder: Dict[str, VertexAIEmbedder] = {}


def get_clip_embedder() -> CLIPEmbedder:
    """Get or create global CLIP embedder instance"""
    if 'instance' not in _clip_embedder:
        _clip_embedder['instance'] = CLIPEmbedder()
    return _clip_embedder['instance']


def get_vertex_embedder() -> VertexAIEmbedder:
    """Get or create global Vertex AI embedder instance"""
    if 'instance' not in _vertex_embedder:
        _vertex_embedder['instance'] = VertexAIEmbedder()
    return _vertex_embedder['instance']


def get_image_embedder() -> ImageEmbedderInterface:
    """
    Get the active image embedder based on configuration.
    
    Returns:
        VertexAIEmbedder if USE_VERTEX_AI_EMBEDDINGS=true
        CLIPEmbedder otherwise (default)
    """
    if settings.use_vertex_ai_embeddings:
        return get_vertex_embedder()
    return get_clip_embedder()


# =====================================================
# GEMINI TEXT EMBEDDINGS (768 dimensions)
# For text/tickets index
# =====================================================

_gemini_client: Dict[str, genai.Client] = {}


def get_gemini_embed_client() -> genai.Client:
    """Get or create Gemini client for embeddings"""
    if 'instance' not in _gemini_client:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not configured for embeddings")
        _gemini_client['instance'] = genai.Client(api_key=settings.gemini_api_key)
        logger.info("Gemini embedding client initialized")
    return _gemini_client['instance']


def embed_text_gemini(text: str) -> List[float]:
    """
    Generate text embeddings using Gemini's text-embedding model.
    Produces 768-dimensional vectors for the tickets index.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector as list (768 dimensions)
    """
    try:
        client = get_gemini_embed_client()
        
        # Use Gemini's embedding model
        # text-embedding-004 produces 768-dim vectors by default
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        
        # Extract embedding vector
        if hasattr(result, 'embeddings') and result.embeddings:
            embedding = result.embeddings[0].values
            logger.debug(f"Generated Gemini embedding with {len(embedding)} dimensions")
            return list(embedding)
        else:
            logger.error("No embeddings returned from Gemini")
            return [0.0] * 768
            
    except Exception as e:
        logger.error(f"Failed to generate Gemini embedding: {e}", exc_info=True)
        return [0.0] * 768


def embed_text(text: str) -> List[float]:
    """
    Generate text embeddings using Gemini (768 dimensions).
    This is the main function used for text/ticket queries.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector as list (768 dimensions)
    """
    return embed_text_gemini(text)


# =====================================================
# IMAGE EMBEDDING FUNCTIONS (512 dimensions)
# Uses CLIP (dev) or Vertex AI (prod) based on config
# =====================================================

def embed_text_clip(text: str) -> List[float]:
    """
    Generate text embeddings for image search (512 dimensions).
    Uses the active embedder (CLIP or Vertex AI).
    
    Note: With Vertex AI, text embeddings are in the SAME space as images,
    enabling more accurate text-to-image search.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector as list (512 dimensions)
    """
    try:
        embedder = get_image_embedder()
        embedding = embedder.embed_text(text)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Failed to embed text for image search: {e}")
        return [0.0] * 512


def embed_image(image_source: Union[str, Path]) -> List[float]:
    """
    Embed an image from path or URL.
    Uses the active embedder (CLIP or Vertex AI) based on config.
    
    Args:
        image_source: File path or HTTP URL
        
    Returns:
        Embedding vector as list (512 dimensions)
    """
    try:
        embedder = get_image_embedder()
        image_str = str(image_source)
        
        if image_str.startswith('http://') or image_str.startswith('https://'):
            embedding = embedder.embed_image_from_url(image_str)
        else:
            embedding = embedder.embed_image_from_path(image_source)
        
        return embedding.tolist()
        
    except Exception as e:
        logger.error(f"Failed to embed image: {e}")
        # Fallback to CLIP if Vertex AI fails
        if settings.use_vertex_ai_embeddings:
            logger.warning("Vertex AI failed, falling back to CLIP")
            try:
                clip = get_clip_embedder()
                if image_str.startswith('http://') or image_str.startswith('https://'):
                    return clip.embed_image_from_url(image_str).tolist()
                else:
                    return clip.embed_image_from_path(image_source).tolist()
            except Exception as fallback_error:
                logger.error(f"CLIP fallback also failed: {fallback_error}")
        return [0.0] * 512


def embed_image_for_search(text_query: str) -> List[float]:
    """
    Generate embedding for text-to-image search.
    Alias for embed_text_clip for clarity.
    
    Args:
        text_query: Search query text
        
    Returns:
        Embedding vector as list (512 dimensions)
    """
    return embed_text_clip(text_query)
