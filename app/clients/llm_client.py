"""
LLM Client for Gemini API Calls
Handles structured LLM requests with JSON responses
"""

import logging
import json
from typing import Dict, Any, Optional
from google import genai
from google.genai import types

from app.config.settings import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for Google Gemini LLM API.
    Handles text generation with support for JSON responses.
    """
    
    def __init__(self):
        """Initialize Gemini LLM client"""
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not configured")
        
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model_name = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        
        logger.info(f"LLM client initialized with model: {self.model_name}")
    
    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Any:
        """
        Call the LLM with prompts
        
        Args:
            system_prompt: System/instruction prompt
            user_prompt: User query/content
            response_format: If "json", expects and parses JSON response
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Parsed JSON dict if response_format="json", otherwise raw text
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        # Combine prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        logger.debug(f"Calling LLM (temperature={temp}, max_tokens={max_tok})...")
        
        try:
            # Build config
            config = types.GenerateContentConfig(
                temperature=temp,
                max_output_tokens=max_tok,
                top_p=0.95,
            )
            
            # If JSON format requested, add instruction
            if response_format == "json":
                config.response_mime_type = "application/json"
            
            # Generate content
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=config
            )
            
            # Extract text - handle various response formats
            response_text = ""
            if hasattr(response, 'text') and response.text:
                response_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                # Try to get text from candidates
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    response_text = part.text
                                    break
                    if response_text:
                        break
            
            # Safety check - ensure we have actual content
            if not response_text or response_text.strip() == "":
                logger.warning(f"LLM returned empty response, raw: {response}")
                if response_format == "json":
                    return {}
                return ""
            
            # Parse JSON if requested
            if response_format == "json":
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.error(f"Raw response: {response_text}")
                    # Return a safe default
                    return {}
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            
            # Return safe defaults
            if response_format == "json":
                return {}
            return f"Error: {str(e)}"
    
    def generate_with_context(
        self,
        system_prompt: str,
        context: str,
        query: str,
        response_format: Optional[str] = None
    ) -> Any:
        """
        Generate response with context and query
        
        Args:
            system_prompt: System instructions
            context: Retrieved context/knowledge
            query: User question/ticket
            response_format: If "json", expects JSON response
            
        Returns:
            LLM response (parsed JSON or text)
        """
        user_prompt = f"""CONTEXT:
{context}

QUERY:
{query}
"""
        
        return self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=response_format
        )


# Global client instance
_client: Dict[str, LLMClient] = {}


def get_llm_client() -> LLMClient:
    """Get or create global LLM client instance"""
    if 'instance' not in _client:
        _client['instance'] = LLMClient()
    return _client['instance']


def call_llm(
    system_prompt: str,
    user_prompt: str,
    response_format: Optional[str] = None,
    temperature: Optional[float] = None
) -> Any:
    """
    Convenience function to call LLM
    
    Args:
        system_prompt: System instructions
        user_prompt: User content
        response_format: "json" for JSON response
        temperature: Override default temperature
        
    Returns:
        LLM response
    """
    client = get_llm_client()
    return client.call_llm(system_prompt, user_prompt, response_format, temperature)
