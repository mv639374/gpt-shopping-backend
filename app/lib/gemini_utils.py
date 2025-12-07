"""
Shared Gemini Processing Utilities
"""
import os
import logging
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# CORRECT IMPORTS - Using the SDK you already have installed
import google.generativeai as genai
from google.generativeai.types import GenerateContentConfig, HarmCategory, HarmBlockThreshold
import google.api_core.exceptions

logger = logging.getLogger(__name__)

# ============================================================================
# SHORTENED SYSTEM PROMPT
# ============================================================================
GEMINI_SYSTEM_PROMPT = """You are a brand extraction AI. Extract brand mentions and citations from AI responses.

OUTPUT: Valid JSON only. No markdown, no explanations.

SCHEMA:
{
  "brands": [
    {
      "name": "Brand Name",
      "domain": "brand.com",
      "position": 1,
      "mention_snippet": ["quote from answer"],
      "overall_sentiment": "Positive|Neutral|Negative",
      "positive_sentiments": ["phrase"],
      "negative_sentiments": ["phrase"],
      "source_citation_url": ["https://url.com"],
      "positive_sentiment_citations": ["https://url.com"],
      "negative_sentiment_citations": ["https://url.com"]
    }
  ],
  "citations": [
    {
      "citation_title": "Page Title",
      "citation_url": "https://url.com",
      "citation_domain": "domain.com",
      "text": "relevant snippet from answer",
      "utm_source": "chatgpt",
      "is_alive": true,
      "description": "brief description",
      "domain_category": "Company|UGC|High trusted media|3rd party media|marketplace",
      "related_brands": ["Brand1", "Brand2"]
    }
  ]
}

RULES:
1. Extract TOP 10 brands max (most prominent first)
2. Use empty arrays [] for missing data
3. Set utm_source to platform name
4. domain_category: Company/UGC/High trusted media/3rd party media/marketplace
5. Keep text under 200 chars
6. Extract domain from URLs
7. Match brands to their citation URLs

STOP after JSON. No extra text."""


class ProhibitedContentError(Exception):
    """Custom exception for content blocked by Gemini"""
    pass


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_citations_for_input(response_data: Dict, platform: str) -> List[Dict[str, Any]]:
    """Extract and clean citations - only essential fields"""
    cited_citations = []
    uncited_citations = []
    
    # Platform-specific citation extraction
    raw_citations = []
    if platform in ["chatgpt"]:
        raw_citations = response_data.get("citations", []) or response_data.get("links_attached", [])
    elif platform in ["perplexity"]:
        raw_citations = response_data.get("sources", [])
    elif platform in ["grok"]:
        raw_citations = response_data.get("citations", [])
    elif platform in ["copilot", "ms_copilot"]:
        raw_citations = response_data.get("sources", [])
    elif platform in ["gemini", "google_gemini"]:
        raw_citations = response_data.get("citations", [])
    elif platform in ["google_ai_mode", "googleaimode"]:
        raw_citations = response_data.get("citations", [])
    else:
        raw_citations = response_data.get("citations", []) or response_data.get("sources", []) or response_data.get("links_attached", [])
    
    if not isinstance(raw_citations, list):
        raw_citations = []
    
    for idx, citation in enumerate(raw_citations):
        if not isinstance(citation, dict):
            continue
        
        cleaned = {}
        url = citation.get("url") or citation.get("citation_url")
        if not url:
            continue
        
        cleaned["url"] = url
        
        title = citation.get("title") or citation.get("citation_title") or citation.get("text", "")
        if title:
            cleaned["title"] = title[:200]
        
        position = citation.get("position")
        if position is not None:
            cleaned["position"] = position
        else:
            cleaned["position"] = idx + 1
        
        is_cited = citation.get("cited", None)
        
        if is_cited is True:
            cited_citations.append(cleaned)
        else:
            uncited_citations.append(cleaned)
    
    return cited_citations + uncited_citations


def detect_ai_platform(response_data: Dict) -> str:
    """Detect platform from response data"""
    if "platform" in response_data:
        return response_data["platform"].lower()
    
    config = response_data.get("config", {})
    if isinstance(config, dict) and "ai_model" in config:
        model = config["ai_model"].lower()
        if "gpt" in model:
            return "chatgpt"
        elif "gemini" in model:
            return "gemini"
        elif "claude" in model:
            return "claude"
    
    return "unknown"


def fix_broken_json(text: str) -> str:
    """Attempt to fix common JSON issues"""
    lines = text.split('\n')
    valid_lines = []
    
    for line in lines:
        if line.strip().endswith('"'):
            if line.count('"') % 2 != 0:
                continue
        valid_lines.append(line)
    
    fixed_text = '\n'.join(valid_lines)
    
    open_braces = fixed_text.count('{')
    close_braces = fixed_text.count('}')
    open_brackets = fixed_text.count('[')
    close_brackets = fixed_text.count(']')
    
    fixed_text += ']' * (open_brackets - close_brackets)
    fixed_text += '}' * (open_braces - close_braces)
    
    return fixed_text


def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON from text with multiple fallback strategies"""
    if not text or not text.strip():
        return None
    
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract from markdown
    json_match = re.search(r'``````', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Extract between { and }
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            json_str = text[first_brace:last_brace + 1]
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Fix broken JSON
    try:
        fixed_text = fix_broken_json(text)
        return json.loads(fixed_text)
    except json.JSONDecodeError:
        pass
    
    return None


# ============================================================================
# MAIN GEMINI API CALL
# ============================================================================

def call_gemini_api(
    model,  # Pass the actual model object
    response_data: Dict,
    platform: str = None,
    max_retries: int = 3
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Call Gemini API with comprehensive logging
    
    Args:
        model: Configured Gemini model object
        response_data: Raw response data to process
        platform: Platform name (chatgpt, perplexity, etc.)
        max_retries: Maximum retry attempts
    
    Returns: (structured_response, token_data)
    """
    if not platform:
        platform = detect_ai_platform(response_data)
    
    # Prepare input
    answer_text = response_data.get("answer_text", "") or response_data.get("response", "")
    citations = clean_citations_for_input(response_data, platform)
    
    input_data = {
        "platform": platform,
        "answer_text": answer_text[:15000],
        "citations": citations[:50]
    }
    
    input_json_str = json.dumps(input_data, ensure_ascii=False, indent=2)
    
    logger.info("=" * 80)
    logger.info(f"📥 GEMINI INPUT (Platform: {platform})")
    logger.info("=" * 80)
    logger.info(f"Answer text length: {len(answer_text)} chars")
    logger.info(f"Citations count: {len(citations)}")
    logger.info(f"Input JSON Preview:\n{input_json_str[:1000]}...")
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"🤖 Gemini API call attempt {attempt}/{max_retries}")
            
            response = model.generate_content(
                input_json_str,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=8000,
                )
            )
            
            raw_output = response.text if hasattr(response, 'text') else str(response)
            
            logger.info("=" * 80)
            logger.info(f"📤 GEMINI RAW OUTPUT (Attempt {attempt})")
            logger.info("=" * 80)
            logger.info(f"Output length: {len(raw_output)} chars")
            logger.info(f"Raw output preview:\n{raw_output[:500]}...")
            
            # Try to parse JSON
            structured_data = extract_json_from_text(raw_output)
            
            if not structured_data:
                logger.error(f"❌ Failed to extract valid JSON from output (Attempt {attempt})")
                if attempt < max_retries:
                    logger.info(f"⏳ Retrying in {2 * attempt}s...")
                    import time
                    time.sleep(2 * attempt)
                continue
            
            # Validate structure
            if "brands" not in structured_data or "citations" not in structured_data:
                logger.error(f"❌ Invalid JSON structure (Attempt {attempt})")
                if attempt < max_retries:
                    import time
                    time.sleep(2 * attempt)
                continue
            
            # Extract token usage
            token_data = {
                "input_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
                "output_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0) if hasattr(response, 'usage_metadata') else 0
            }
            
            logger.info("=" * 80)
            logger.info(f"✅ GEMINI SUCCESS (Attempt {attempt})")
            logger.info("=" * 80)
            logger.info(f"📊 Brands extracted: {len(structured_data.get('brands', []))}")
            logger.info(f"📊 Citations extracted: {len(structured_data.get('citations', []))}")
            logger.info(f"📊 Token usage: {token_data['input_tokens']} in, {token_data['output_tokens']} out")
            
            return structured_data, token_data
            
        except google.api_core.exceptions.GoogleAPIError as e:
            logger.error(f"❌ Gemini API error on attempt {attempt}: {str(e)}")
            if "SAFETY" in str(e).upper() or "PROHIBITED" in str(e).upper():
                raise ProhibitedContentError(f"Content blocked: {e}")
            if attempt < max_retries:
                import time
                time.sleep(2 * attempt)
        except Exception as e:
            logger.error(f"❌ Unexpected error on attempt {attempt}: {str(e)}")
            if attempt < max_retries:
                import time
                time.sleep(2 * attempt)
    
    logger.warning(f"⚠️ All {max_retries} attempts failed, returning empty structure")
    return {"brands": [], "citations": []}, {"input_tokens": 0, "output_tokens": 0}
