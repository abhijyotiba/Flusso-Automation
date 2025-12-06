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
        
        logger.info(f"LLM client initialized with model: {self.model_name}, max_tokens: {self.max_tokens}")
    
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
        
        logger.info(f"📤 LLM Request: model={self.model_name}, temperature={temp}, max_tokens={max_tok}")
        logger.debug(f"📤 Prompt length: {len(full_prompt)} chars")
        
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
            
            # === DETAILED RESPONSE DEBUGGING ===
            finish_reason = None
            token_count = None
            
            # Check for finish reason and token usage
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    logger.info(f"📥 LLM finish_reason: {finish_reason}")
                    
                    # Check if response was truncated
                    if str(finish_reason).upper() in ['MAX_TOKENS', 'LENGTH', 'STOP_LIMIT']:
                        logger.warning(f"⚠️ LLM RESPONSE TRUNCATED! finish_reason={finish_reason}. Consider increasing max_tokens (current: {max_tok})")
                
                # Try to get token count
                if hasattr(candidate, 'token_count'):
                    token_count = candidate.token_count
                    logger.info(f"📥 LLM tokens used: {token_count}")
            
            # Check usage metadata if available
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                if usage:
                    prompt_tokens = getattr(usage, 'prompt_token_count', 'N/A')
                    output_tokens = getattr(usage, 'candidates_token_count', 'N/A')
                    total_tokens = getattr(usage, 'total_token_count', 'N/A')
                    logger.info(f"📊 Token usage: prompt={prompt_tokens}, output={output_tokens}, total={total_tokens}")
                    
                    # Warn if output tokens is close to max
                    if isinstance(output_tokens, int) and output_tokens >= max_tok * 0.95:
                        logger.warning(f"⚠️ Output tokens ({output_tokens}) is at/near max_tokens limit ({max_tok})! Response likely truncated!")
            
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
            
            # Log response details
            logger.info(f"📥 LLM Response: {len(response_text)} chars received")
            logger.debug(f"📥 Response preview: {response_text[:200]}..." if len(response_text) > 200 else f"📥 Full response: {response_text}")
            
            # Safety check - ensure we have actual content
            if not response_text or response_text.strip() == "":
                logger.warning(f"⚠️ LLM returned empty response!")
                logger.warning(f"Raw response object: {response}")
                if response_format == "json":
                    return {}
                return ""
            
            # Check for incomplete responses (missing expected sections)
            expected_sections = ["## 🎫 TICKET ANALYSIS", "## 🔧 PRODUCT IDENTIFICATION", "## 💡 SUGGESTED ACTIONS", "## 📝 SUGGESTED RESPONSE"]
            missing_sections = [s for s in expected_sections if s not in response_text]
            if missing_sections and response_format != "json":
                logger.warning(f"⚠️ Response may be incomplete! Missing sections: {missing_sections}")
                logger.warning(f"Response ends with: ...{response_text[-100:]}" if len(response_text) > 100 else f"Full response: {response_text}")
            
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
            logger.error(f"❌ Error calling LLM: {e}", exc_info=True)
            
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
