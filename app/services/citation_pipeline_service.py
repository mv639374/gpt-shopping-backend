"""
Citation Pipeline Service with rollback and comprehensive logging
NO automatic cleanup - only rollback on failure
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import google.generativeai as genai
import google.api_core.exceptions

from app.core.config import settings
from app.utils.db import get_supabase_client
from app.core.logger import logger
from app.utils.logging_utils import create_batch_logger, close_batch_logger

# ============================================================================
# GEMINI SYSTEM PROMPT
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

class GeminiRateLimitError(Exception):
    """Custom exception for rate limit errors"""
    pass

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_citations_for_input(response_data: Dict, platform: str) -> List[Dict]:
    """Extract and clean citations"""
    cited_citations = []
    uncited_citations = []
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
    if isinstance(response_data, dict):
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

def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON from text"""
    if not text or not text.strip():
        return None
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    json_match = re.search(r'``````', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            json_str = text[first_brace:last_brace + 1]
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    return None

def call_gemini_api(model, response_data: Dict, platform: str = None) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Call Gemini API with exponential backoff retry logic
    """
    # Get retry configuration from environment
    max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "5"))
    initial_delay = float(os.getenv("GEMINI_INITIAL_RETRY_DELAY", "2"))
    max_delay = float(os.getenv("GEMINI_MAX_RETRY_DELAY", "60"))
    
    if not platform:
        platform = detect_ai_platform(response_data)
    
    # Handle both dict and string response data
    if isinstance(response_data, str):
        answer_text = response_data
    else:
        answer_text = response_data.get("answer_text", "") or response_data.get("response", "")
    
    citations = clean_citations_for_input(response_data if isinstance(response_data, dict) else {}, platform)
    
    input_data = {
        "platform": platform,
        "answer_text": answer_text[:15000],
        "citations": citations[:50]
    }
    
    prompt = f"{GEMINI_SYSTEM_PROMPT}\n\nINPUT DATA:\n{json.dumps(input_data, ensure_ascii=False, indent=2)}"
    
    logger.info("=" * 80)
    logger.info(f"📥 GEMINI INPUT (Platform: {platform})")
    logger.info(f"Answer text length: {len(answer_text)} chars")
    logger.info(f"Citations count: {len(citations)}")
    logger.info(f"Max retries: {max_retries}, Initial delay: {initial_delay}s")
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"🤖 Gemini API call attempt {attempt}/{max_retries}")
            response = model.generate_content(prompt)
            
            raw_output = response.text if hasattr(response, 'text') else str(response)
            logger.info(f"📤 GEMINI RAW OUTPUT (length: {len(raw_output)} chars)")
            
            structured_data = extract_json_from_text(raw_output)
            
            if not structured_data:
                logger.error(f"❌ Failed to extract JSON (Attempt {attempt})")
                if attempt < max_retries:
                    delay = min(initial_delay * (2 ** (attempt - 1)), max_delay)
                    logger.info(f"⏳ Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
            
            if "brands" not in structured_data or "citations" not in structured_data:
                logger.error(f"❌ Invalid JSON structure (Attempt {attempt})")
                if attempt < max_retries:
                    delay = min(initial_delay * (2 ** (attempt - 1)), max_delay)
                    logger.info(f"⏳ Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
            
            token_data = {
                "input_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
                "output_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0) if hasattr(response, 'usage_metadata') else 0
            }
            
            logger.info(f"✅ SUCCESS: {len(structured_data.get('brands', []))} brands, {len(structured_data.get('citations', []))} citations")
            logger.info(f"📊 Tokens: {token_data['input_tokens']} in, {token_data['output_tokens']} out")
            
            return structured_data, token_data
        
        except google.api_core.exceptions.ResourceExhausted as e:
            # 429 Rate Limit Error
            logger.error(f"❌ Rate limit error (Attempt {attempt}): {str(e)}")
            if attempt < max_retries:
                delay = min(initial_delay * (2 ** (attempt - 1)), max_delay)
                logger.warning(f"⏳ Rate limited. Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
            else:
                raise GeminiRateLimitError(f"Rate limit exceeded after {max_retries} attempts")
        
        except google.api_core.exceptions.GoogleAPIError as e:
            logger.error(f"❌ Gemini API error (Attempt {attempt}): {str(e)}")
            if "SAFETY" in str(e).upper() or "PROHIBITED" in str(e).upper():
                raise ProhibitedContentError(f"Content blocked: {e}")
            if attempt < max_retries:
                delay = min(initial_delay * (2 ** (attempt - 1)), max_delay)
                logger.info(f"⏳ Retrying in {delay:.1f}s...")
                time.sleep(delay)
        
        except Exception as e:
            logger.error(f"❌ Error (Attempt {attempt}): {str(e)}")
            if attempt < max_retries:
                delay = min(initial_delay * (2 ** (attempt - 1)), max_delay)
                logger.info(f"⏳ Retrying in {delay:.1f}s...")
                time.sleep(delay)
    
    logger.warning(f"⚠️ All {max_retries} attempts failed")
    return {"brands": [], "citations": []}, {"input_tokens": 0, "output_tokens": 0}

# ============================================================================
# SERVICE CLASS
# ============================================================================

class GeminiCitationService:
    """Service for processing citations using Gemini API"""
    
    def __init__(self, supabase_client, schema_name: str = "public"):
        """Initialize service with Supabase client"""
        self.supabase = supabase_client
        self.schema_name = schema_name
        self.model = self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Gemini model"""
        try:
            api_key = settings.gemini_api_key
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment")
            
            genai.configure(api_key=api_key)
            
            model = genai.GenerativeModel(
                model_name=os.getenv("RESPONSE_PROCESSING_MODEL", "gemini-2.0-flash-exp"),
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=8000,
                )
            )
            
            logger.info("✅ Gemini model initialized")
            return model
        
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini: {e}")
            return None
    
    def process_single_response(self, response_record: Dict) -> Dict:
        """Process a single response and save to database with rollback on failure"""
        response_id = response_record['id']
        logger.info(f"🔄 Processing response {response_id[:8]}...")
        
        response_data = response_record.get('response_data', {})
        if isinstance(response_data, str):
            try:
                response_data = json.loads(response_data)
            except json.JSONDecodeError:
                response_data = {"answer_text": response_data}
        
        platform = detect_ai_platform(response_data)
        
        # Track what we've saved for rollback
        saved_ids = {
            "structured_id": None,
            "brand_ids": [],
            "mention_ids": [],
            "citation_ids": []
        }
        
        try:
            # Call Gemini API
            structured_data, token_data = call_gemini_api(
                self.model,
                response_data,
                platform=platform
            )
        
        except (ProhibitedContentError, GeminiRateLimitError) as e:
            error_type = "content_blocked" if isinstance(e, ProhibitedContentError) else "rate_limit"
            logger.error(f"❌ {error_type} for {response_id[:8]}: {e}")
            return {
                "response_id": response_id,
                "success": False,
                "error": error_type,
                "brands_count": 0,
                "mentions_created": 0,
                "citations_created": 0
            }
        
        brands = structured_data.get("brands", [])
        citations = structured_data.get("citations", [])
        
        valid_brands = [b for b in brands if b.get("name") and b.get("domain")]
        valid_citations = [c for c in citations if c.get("citation_url") and c.get("citation_domain")]
        
        logger.info(f"🔍 Validated: {len(valid_brands)} brands, {len(valid_citations)} citations")
        
        try:
            # Save structured response
            saved_ids["structured_id"] = self._save_structured_response(response_id, structured_data, token_data)
            
            # Save brands and mentions
            brand_result = self._save_brands_and_mentions(response_id, valid_brands)
            saved_ids["brand_ids"] = brand_result["brand_ids"]
            saved_ids["mention_ids"] = brand_result["mention_ids"]
            
            # Save citations
            saved_ids["citation_ids"] = self._save_citations(response_id, valid_citations)
            
            logger.info(f"✅ Successfully saved data for {response_id[:8]}")
            
            return {
                "response_id": response_id,
                "success": True,
                "brands_count": len(valid_brands),
                "mentions_created": len(saved_ids["mention_ids"]),
                "citations_created": len(saved_ids["citation_ids"])
            }
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Database error for {response_id[:8]}: {error_msg}")
            
            # Rollback all saved data
            self._rollback_saved_data(response_id, saved_ids)
            
            if "duplicate key" in error_msg.lower():
                logger.warning(f"⚠️ Duplicate entry detected for {response_id[:8]}")
                return {
                    "response_id": response_id,
                    "success": False,
                    "error": "duplicate_entry",
                    "brands_count": 0,
                    "mentions_created": 0,
                    "citations_created": 0
                }
            
            # Re-raise for other errors
            raise
    
    def _save_structured_response(self, response_id: str, structured_data: Dict, token_data: Dict) -> str:
        """Save structured response data"""
        try:
            data = {
                "prompt_response_id": response_id,
                "metadata": {
                    "token_usage": token_data,
                    "processing_time": datetime.utcnow().isoformat()
                },
                "structured_data": structured_data
            }
            
            result = self.supabase.table("prompts_responses_structured").insert(data).execute()
            
            if not result.data:
                raise Exception("Failed to save structured response")
            
            return result.data[0]["id"]
        
        except Exception as e:
            logger.error(f"❌ Error saving structured response: {e}")
            raise
    
    def _save_brands_and_mentions(self, response_id: str, brands: List[Dict]) -> Dict[str, List[str]]:
        """Save brands and their mentions"""
        brand_ids = []
        mention_ids = []
        
        for brand in brands:
            try:
                # Save brand
                brand_data = {
                    "name": brand["name"],
                    "domain": brand["domain"],
                    "metadata": {
                        "first_seen": datetime.utcnow().isoformat(),
                        "source_response_id": response_id
                    }
                }
                
                # Upsert brand
                result = self.supabase.table("brands").upsert(
                    brand_data,
                    on_conflict="name,domain"
                ).execute()
                
                if not result.data:
                    logger.warning(f"⚠️ Failed to upsert brand: {brand['name']}")
                    continue
                
                brand_id = result.data[0]["id"]
                brand_ids.append(brand_id)
                
                # Save brand mention
                mention_data = {
                    "prompt_response_id": response_id,
                    "brand_id": brand_id,
                    "position": brand.get("position", 0),
                    "mention_snippet": brand.get("mention_snippet", []),
                    "overall_sentiment": brand.get("overall_sentiment", "Neutral").lower(),
                    "positive_sentiments": brand.get("positive_sentiments", []),
                    "negative_sentiments": brand.get("negative_sentiments", []),
                    "source_citation_urls": brand.get("source_citation_url", []),
                    "positive_sentiment_citations": brand.get("positive_sentiment_citations", []),
                    "negative_sentiment_citations": brand.get("negative_sentiment_citations", [])
                }
                
                result = self.supabase.table("brand_mentions").insert(mention_data).execute()
                
                if result.data:
                    mention_ids.append(result.data[0]["id"])
                
            except Exception as e:
                logger.error(f"❌ Error processing brand {brand.get('name')}: {e}")
                continue
        
        return {
            "brand_ids": brand_ids,
            "mention_ids": mention_ids
        }
    
    def _save_citations(self, response_id: str, citations: List[Dict]) -> List[str]:
        """Save citations"""
        citation_ids = []
        
        for citation in citations:
            try:
                # Prepare citation data
                citation_data = {
                    "prompt_response_id": response_id,
                    "citation_title": citation.get("citation_title", "")[:500],
                    "citation_url": citation["citation_url"],
                    "citation_domain": citation["citation_domain"],
                    "text": citation.get("text", "")[:1000],
                    "utm_source": citation.get("utm_source", ""),
                    "is_alive": citation.get("is_alive", True),
                    "description": citation.get("description", "")[:500],
                    "domain_category": citation.get("domain_category", ""),
                    "related_brands": citation.get("related_brands", []),
                    "position": citation.get("position", 0)
                }
                
                # Insert citation
                result = self.supabase.table("citations").insert(citation_data).execute()
                
                if result.data:
                    citation_ids.append(result.data[0]["id"])
                
            except Exception as e:
                logger.error(f"❌ Error saving citation {citation.get('citation_url')}: {e}")
                continue
        
        return citation_ids
    
    def _rollback_saved_data(self, response_id: str, saved_ids: Dict[str, List[str]]) -> None:
        """Rollback all saved data for a response"""
        logger.warning(f"🔄 Rolling back data for {response_id[:8]}...")
        
        try:
            # Delete citations
            if saved_ids["citation_ids"]:
                self.supabase.table("citations").delete().in_("id", saved_ids["citation_ids"]).execute()
            
            # Delete brand mentions
            if saved_ids["mention_ids"]:
                self.supabase.table("brand_mentions").delete().in_("id", saved_ids["mention_ids"]).execute()
            
            # Delete structured response
            if saved_ids["structured_id"]:
                self.supabase.table("prompts_responses_structured").delete().eq("id", saved_ids["structured_id"]).execute()
            
            # Note: We don't delete brands as they might be referenced by other records
            
            logger.info(f"✅ Successfully rolled back data for {response_id[:8]}")
        
        except Exception as e:
            logger.error(f"❌ Error during rollback for {response_id[:8]}: {e}")
            # Continue with the original error

# ✅ Global job tracking (use Redis in production)
batch_jobs: Dict[str, Dict[str, Any]] = {}

class CitationPipelineService:
    """Service to process responses through Gemini and populate citation tables"""
    
    def __init__(self):
        self.supabase = get_supabase_client()
        self.gemini_service = GeminiCitationService(
            supabase_client=self.supabase,
            schema_name="public"
        )
    
    async def get_unprocessed_responses(self) -> List[Dict[str, Any]]:
        """
        Get responses from responses-2 that haven't been processed yet
        (no corresponding entry in prompts_responses_structured)
        """
        try:
            # Get all response IDs from responses-2
            all_responses = self.supabase.table("responses-2")\
                .select("id, response, prompt_id, created_at, region, platform, model_name")\
                .execute()
            
            if not all_responses.data:
                logger.info("📭 No responses found in responses-2 table")
                return []
            
            # Get already processed response IDs
            processed = self.supabase.table("prompts_responses_structured")\
                .select("prompt_response_id")\
                .execute()
            
            processed_ids = {item["prompt_response_id"] for item in (processed.data or [])}
            
            # Filter unprocessed
            unprocessed = [
                resp for resp in all_responses.data
                if resp["id"] not in processed_ids and resp.get("response")
            ]
            
            logger.info(
                f"📊 Found {len(unprocessed)} unprocessed responses out of {len(all_responses.data)} total"
            )
            
            return unprocessed
        
        except Exception as e:
            logger.error(f"❌ Error fetching unprocessed responses: {e}")
            raise
    
    async def process_single_response(self, response_data: Dict[str, Any], batch_logger=None) -> Dict[str, Any]:
        """Process a single response through the pipeline with rollback on failure"""
        log = batch_logger if batch_logger else logger
        response_id = response_data["id"]
        
        result = {
            "response_id": response_id,
            "status": "pending",
            "error": None,
            "brands_extracted": 0,
            "mentions_created": 0,
            "citations_created": 0,
            "rolled_back": False,
            "token_usage": {"input_tokens": 0, "output_tokens": 0}
        }
        
        try:
            log.info(f"🔄 Processing response {response_id[:8]}...")
            
            # Pass the 'response' field as response_data
            response_record = {
                "id": response_id,
                "response_data": response_data.get("response", "")
            }
            
            gemini_result = self.gemini_service.process_single_response(response_record)
            
            if gemini_result["success"]:
                result["status"] = "success"
                result["brands_extracted"] = gemini_result.get("brands_count", 0)
                result["mentions_created"] = gemini_result.get("mentions_count", 0)
                result["citations_created"] = gemini_result.get("citations_count", 0)
                log.info(
                    f"✅ Success: {result['brands_extracted']} brands, "
                    f"{result['mentions_created']} mentions, {result['citations_created']} citations"
                )
            else:
                error_type = gemini_result.get("error", "Unknown error")
                result["status"] = "failed"
                result["error"] = error_type
                result["rolled_back"] = gemini_result.get("rolled_back", False)
                
                if error_type == "duplicate_entry":
                    log.warning(f"⚠️ Skipped (already processed): {response_id[:8]}")
                    result["status"] = "skipped"
                else:
                    log.error(f"❌ Failed: {result['error']}")
                    if result["rolled_back"]:
                        log.warning(f"🔄 Data rolled back for {response_id[:8]}")
            
            return result
        
        except Exception as e:
            error_msg = str(e)
            log.error(f"❌ Error processing response {response_id[:8]}: {error_msg}")
            result["status"] = "failed"
            result["error"] = error_msg
            return result
    
    async def process_batch(self, batch_size: int = 10, max_concurrent: int = 3) -> Dict[str, Any]:
        """
        Process a batch of responses with concurrency control
        NO cleanup - only rollback on individual failures
        """
        batch_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        batch_logger = create_batch_logger(batch_id)
        
        try:
            batch_logger.info(f"📊 Fetching unprocessed responses...")
            unprocessed = await self.get_unprocessed_responses()
            
            if not unprocessed:
                batch_logger.info("✅ No unprocessed responses found")
                close_batch_logger(batch_logger)
                return {
                    "total_found": 0,
                    "processed": 0,
                    "successful": 0,
                    "failed": 0,
                    "skipped": 0,
                    "rolled_back": 0,
                    "remaining": 0,
                    "details": [],
                    "log_file": f"batch_{batch_id}.log"
                }
            
            to_process = unprocessed[:batch_size]
            batch_logger.info(f"🚀 Starting batch processing of {len(to_process)} responses (max_concurrent={max_concurrent})...")
            batch_logger.info(f"📋 Response IDs: {[r['id'][:8] for r in to_process]}")
            
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_with_semaphore_and_logging(response_data, index):
                async with semaphore:
                    batch_logger.info(f"🔄 [{index+1}/{len(to_process)}] Processing response: {response_data['id'][:8]}...")
                    result = await self.process_single_response(response_data, batch_logger)
                    
                    if result["status"] == "success":
                        batch_logger.info(
                            f"✅ [{index+1}/{len(to_process)}] Success: {result['brands_extracted']} brands, "
                            f"{result['mentions_created']} mentions, {result['citations_created']} citations"
                        )
                    elif result["status"] == "skipped":
                        batch_logger.warning(
                            f"⚠️ [{index+1}/{len(to_process)}] Skipped (duplicate): {result['response_id'][:8]}"
                        )
                    else:
                        batch_logger.error(
                            f"❌ [{index+1}/{len(to_process)}] Failed: {result.get('error', 'Unknown error')}"
                        )
                        if result.get("rolled_back"):
                            batch_logger.warning(
                                f"🔄 [{index+1}/{len(to_process)}] Rolled back: {result['response_id'][:8]}"
                            )
                    
                    return result
            
            results = await asyncio.gather(
                *[process_with_semaphore_and_logging(resp, idx) for idx, resp in enumerate(to_process)],
                return_exceptions=True
            )
            
            successful = sum(1 for r in results if isinstance(r, dict) and r["status"] == "success")
            skipped = sum(1 for r in results if isinstance(r, dict) and r["status"] == "skipped")
            failed = sum(1 for r in results if isinstance(r, dict) and r["status"] == "failed")
            rolled_back = sum(1 for r in results if isinstance(r, dict) and r.get("rolled_back", False))
            
            total_brands = sum(r.get("brands_extracted", 0) for r in results if isinstance(r, dict))
            total_mentions = sum(r.get("mentions_created", 0) for r in results if isinstance(r, dict))
            total_citations = sum(r.get("citations_created", 0) for r in results if isinstance(r, dict))
            
            # Get rolled back response IDs
            rolled_back_ids = [r["response_id"][:16] for r in results if isinstance(r, dict) and r.get("rolled_back", False)]
            
            summary = {
                "total_found": len(unprocessed),
                "processed": len(results),
                "successful": successful,
                "failed": failed,
                "skipped": skipped,
                "rolled_back": rolled_back,
                "rolled_back_ids": rolled_back_ids,
                "remaining": len(unprocessed) - len(to_process),
                "total_brands": total_brands,
                "total_mentions": total_mentions,
                "total_citations": total_citations,
                "details": [r for r in results if isinstance(r, dict)],
                "log_file": f"batch_{batch_id}.log"
            }
            
            batch_logger.info("=" * 80)
            batch_logger.info(f"📊 BATCH SUMMARY:")
            batch_logger.info(f"   Total found: {summary['total_found']}")
            batch_logger.info(f"   Processed: {summary['processed']}")
            batch_logger.info(f"   ✅ Successful: {summary['successful']}")
            batch_logger.info(f"   ⚠️  Skipped: {summary['skipped']}")
            batch_logger.info(f"   ❌ Failed: {summary['failed']}")
            batch_logger.info(f"   🔄 Rolled back: {summary['rolled_back']}")
            if rolled_back_ids:
                batch_logger.info(f"   🔄 Rolled back IDs: {', '.join(rolled_back_ids)}")
            batch_logger.info(f"   📋 Remaining: {summary['remaining']}")
            batch_logger.info(f"   🏷️  Total brands: {total_brands}")
            batch_logger.info(f"   📝 Total mentions: {total_mentions}")
            batch_logger.info(f"   🔗 Total citations: {total_citations}")
            batch_logger.info("=" * 80)
            
            logger.info(
                f"📊 Batch {batch_id} complete: {successful} successful, {skipped} skipped, "
                f"{failed} failed, {rolled_back} rolled back, {summary['remaining']} remaining"
            )
            
            close_batch_logger(batch_logger)
            return summary
        
        except Exception as e:
            batch_logger.error(f"❌ Batch processing error: {e}", exc_info=True)
            logger.error(f"❌ Batch processing error: {e}")
            close_batch_logger(batch_logger)
            raise

# Global instance
citation_pipeline = CitationPipelineService()

# ✅ Background task function for processing all responses
async def process_all_responses_background(
    job_id: str,
    batch_size: int,
    max_concurrent: int
):
    """
    Background task to process ALL unprocessed responses
    Processes them in internal batches but continues until all are done
    """
    try:
        # Get total count
        unprocessed = await citation_pipeline.get_unprocessed_responses()
        total_count = len(unprocessed)
        
        if total_count == 0:
            batch_jobs[job_id]["status"] = "completed"
            batch_jobs[job_id]["total"] = 0
            return
        
        # Update job info
        total_batches = (total_count + batch_size - 1) // batch_size
        batch_jobs[job_id].update({
            "total": total_count,
            "total_batches": total_batches,
            "started_at": datetime.utcnow().isoformat(),
            "status": "processing",
            "rolled_back_ids": []
        })
        
        logger.info(f"🚀 Job {job_id}: Processing {total_count} responses in {total_batches} batches")
        
        # Process in batches until all are done
        batch_num = 0
        while True:
            batch_num += 1
            batch_jobs[job_id]["current_batch"] = batch_num
            
            logger.info(f"📦 Job {job_id}: Starting batch {batch_num}/{total_batches}")
            
            # Process one batch
            result = await citation_pipeline.process_batch(
                batch_size=batch_size,
                max_concurrent=max_concurrent
            )
            
            # Update cumulative stats
            batch_jobs[job_id]["processed"] += result["processed"]
            batch_jobs[job_id]["successful"] += result["successful"]
            batch_jobs[job_id]["failed"] += result["failed"]
            batch_jobs[job_id]["skipped"] += result.get("skipped", 0)
            batch_jobs[job_id]["rolled_back_ids"].extend(result.get("rolled_back_ids", []))
            
            # Check if we're done
            if result["remaining"] == 0:
                logger.info(f"✅ Job {job_id}: All responses processed!")
                break
            
            # Safety check
            if batch_num >= total_batches + 5:
                logger.warning(f"⚠️ Job {job_id}: Reached max batch limit, stopping")
                break
        
        # Mark as completed
        batch_jobs[job_id].update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat()
        })
        
        logger.info(
            f"✅ Job {job_id} COMPLETED: "
            f"{batch_jobs[job_id]['successful']} successful, "
            f"{batch_jobs[job_id]['skipped']} skipped, "
            f"{batch_jobs[job_id]['failed']} failed, "
            f"{len(batch_jobs[job_id]['rolled_back_ids'])} rolled back"
        )
    
    except Exception as e:
        logger.error(f"❌ Job {job_id} FAILED: {e}", exc_info=True)
        batch_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat()
        })
