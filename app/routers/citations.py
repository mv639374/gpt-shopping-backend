from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Optional
from app.services.citation_pipeline_service import citation_pipeline, batch_jobs, process_all_responses_background
from app.core.logger import logger
import uuid

router = APIRouter(prefix="/citations", tags=["Citations"])

@router.get("/processing-status")
async def get_processing_status():
    """
    Get current processing status of responses
    
    Returns:
        - total_responses: Total responses in responses-2 table
        - successful: Number of successfully processed responses
        - unprocessed: Number of responses not yet processed
        - failed: Number of failed processing attempts
    """
    try:
        supabase = citation_pipeline.supabase
        
        # Get total responses
        total_response = supabase.table("responses-2").select("id", count="exact").execute()
        total_responses = total_response.count if total_response.count else 0
        
        # Get processed responses (successful)
        processed_response = supabase.table("prompts_responses_structured").select(
            "prompt_response_id, processing_status", 
            count="exact"
        ).execute()
        
        # Count successful vs failed
        successful = 0
        failed = 0
        if processed_response.data:
            for record in processed_response.data:
                if record.get("processing_status") == "success":
                    successful += 1
                elif record.get("processing_status") == "failed":
                    failed += 1
        
        # Calculate unprocessed
        unprocessed = max(0, total_responses - successful - failed)
        
        # Get statistics from brand mentions and citations
        try:
            brands_response = supabase.table("child_brands").select("id", count="exact").execute()
            total_brands = brands_response.count if brands_response.count else 0
            
            mentions_response = supabase.table("response_brand_mentions").select("id", count="exact").execute()
            total_mentions = mentions_response.count if mentions_response.count else 0
            
            citations_response = supabase.table("citations").select("id", count="exact").execute()
            total_citations = citations_response.count if citations_response.count else 0
            
            statistics = {
                "total_brands": total_brands,
                "total_mentions": total_mentions,
                "total_citations": total_citations,
            }
        except Exception as stats_error:
            logger.warning(f"⚠️ Error fetching statistics: {stats_error}")
            statistics = {
                "total_brands": 0,
                "total_mentions": 0,
                "total_citations": 0,
            }
        
        logger.info(
            f"📊 Status: {total_responses} total, {successful} successful, "
            f"{failed} failed, {unprocessed} unprocessed"
        )
        
        return {
            "total_responses": total_responses,
            "successful": successful,
            "unprocessed": unprocessed,
            "failed": failed,
            "statistics": statistics,
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching processing status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-batch")
async def process_batch(
    batch_size: int = Query(10, ge=1, le=100, description="Number of responses to process (1-100)"),
    max_concurrent: int = Query(3, ge=1, le=10, description="Max concurrent Gemini API calls (1-10)")
):
    """
    Process a batch of unprocessed responses through the citation extraction pipeline
    """
    try:
        logger.info(
            f"🚀 Starting batch processing: {batch_size} responses, "
            f"{max_concurrent} concurrent tasks"
        )
        
        result = await citation_pipeline.process_batch(
            batch_size=batch_size,
            max_concurrent=max_concurrent
        )
        
        logger.info(
            f"✅ Batch processing complete: {result['successful']} successful, "
            f"{result['failed']} failed, {result.get('rolled_back', 0)} rolled back, "
            f"{result['remaining']} remaining"
        )
        
        return {
            "message": f"Processed {result['processed']} responses",
            "processed": result["processed"],
            "successful": result["successful"],
            "failed": result["failed"],
            "skipped": result.get("skipped", 0),
            "rolled_back": result.get("rolled_back", 0),
            "rolled_back_ids": result.get("rolled_back_ids", []),
            "remaining": result["remaining"],
            "total_found": result["total_found"],
            "log_file": result.get("log_file")
            # ❌ REMOVED: "cleanup" field
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ✅ NEW: Process ALL unprocessed responses
@router.post("/process-all")
async def process_all_responses(
    background_tasks: BackgroundTasks,
    batch_size: int = Query(50, ge=10, le=100, description="Internal batch size"),
    max_concurrent: int = Query(5, ge=1, le=10, description="Max concurrent Gemini API calls")
):
    """
    Start processing ALL unprocessed responses asynchronously in background
    Returns job_id to track progress
    """
    try:
        job_id = str(uuid.uuid4())
        
        # Initialize job tracking
        batch_jobs[job_id] = {
            "status": "started",
            "total": 0,
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "current_batch": 0,
            "total_batches": 0,
            "started_at": None,
            "completed_at": None,
            "rolled_back_ids": [],
            "error": None
        }
        
        # Add background task
        background_tasks.add_task(
            process_all_responses_background,
            job_id=job_id,
            batch_size=batch_size,
            max_concurrent=max_concurrent
        )
        
        logger.info(f"✅ Started processing all responses - Job ID: {job_id}")
        
        return {
            "job_id": job_id,
            "message": "Processing all unprocessed responses in background",
            "status": "started"
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to start processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/batch-status/{job_id}")
async def get_batch_status(job_id: str):
    """Get the status of a batch processing job"""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return batch_jobs[job_id]

@router.get("/unprocessed-count")
async def get_unprocessed_count():
    """
    Get count of unprocessed responses
    
    Returns:
        Number of responses that haven't been processed yet
    """
    try:
        unprocessed = await citation_pipeline.get_unprocessed_responses()
        
        return {
            "unprocessed_count": len(unprocessed),
            "message": f"{len(unprocessed)} responses waiting to be processed"
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting unprocessed count: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-single/{response_id}")
async def process_single_response(response_id: str):
    """
    Process a single specific response by ID
    
    Args:
        response_id: ID of the response to process
    
    Returns:
        Processing result for the single response
    """
    try:
        supabase = citation_pipeline.supabase
        
        # Get the specific response
        response_data_result = supabase.table("responses-2").select("*").eq("id", response_id).execute()
        
        if not response_data_result.data or len(response_data_result.data) == 0:
            raise HTTPException(status_code=404, detail=f"Response {response_id} not found")
        
        response_data = response_data_result.data[0]
        
        # Check if already processed
        existing = supabase.table("prompts_responses_structured").select("processing_status").eq(
            "prompt_response_id", response_id
        ).execute()
        
        if existing.data and len(existing.data) > 0:
            status = existing.data[0].get("processing_status")
            return {
                "message": f"Response {response_id[:8]}... already processed",
                "status": status,
                "response_id": response_id,
            }
        
        # Process the response
        logger.info(f"🔄 Processing single response: {response_id[:8]}...")
        result = await citation_pipeline.process_single_response(response_data)
        
        logger.info(f"✅ Single response processing complete: {result['status']}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing single response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/reset-failed")
async def reset_failed_responses():
    """
    Reset failed responses to allow reprocessing
    
    Deletes records with processing_status='failed' from prompts_responses_structured
    so they can be reprocessed in the next batch
    
    Returns:
        Number of failed responses reset
    """
    try:
        supabase = citation_pipeline.supabase
        
        # Get failed responses
        failed_responses = supabase.table("prompts_responses_structured").select(
            "prompt_response_id"
        ).eq("processing_status", "failed").execute()
        
        if not failed_responses.data or len(failed_responses.data) == 0:
            return {
                "message": "No failed responses to reset",
                "reset_count": 0
            }
        
        failed_ids = [r["prompt_response_id"] for r in failed_responses.data]
        
        # Delete failed records
        for response_id in failed_ids:
            supabase.table("prompts_responses_structured").delete().eq(
                "prompt_response_id", response_id
            ).execute()
        
        logger.info(f"🔄 Reset {len(failed_ids)} failed responses for reprocessing")
        
        return {
            "message": f"Reset {len(failed_ids)} failed responses",
            "reset_count": len(failed_ids),
            "response_ids": failed_ids[:10] if len(failed_ids) > 10 else failed_ids,
        }
        
    except Exception as e:
        logger.error(f"❌ Error resetting failed responses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/brand-stats")
async def get_brand_statistics():
    """
    Get statistics about extracted brands
    
    Returns:
        - Top brands by mention count
        - Sentiment distribution
        - Citation counts
    """
    try:
        supabase = citation_pipeline.supabase
        
        # Get all brand mentions
        mentions = supabase.table("response_brand_mentions").select(
            "brand_name, overall_sentiment, citations_count"
        ).execute()
        
        if not mentions.data:
            return {
                "total_brands": 0,
                "total_mentions": 0,
                "top_brands": [],
                "sentiment_distribution": {},
            }
        
        # Calculate statistics
        brand_counts = {}
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0}
        
        for mention in mentions.data:
            brand_name = mention.get("brand_name", "Unknown")
            sentiment = mention.get("overall_sentiment", "Neutral")
            
            # Count brand mentions
            brand_counts[brand_name] = brand_counts.get(brand_name, 0) + 1
            
            # Count sentiments
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
        
        # Get top 20 brands
        top_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Get unique brand count
        unique_brands = supabase.table("child_brands").select("id", count="exact").execute()
        total_unique_brands = unique_brands.count if unique_brands.count else 0
        
        return {
            "total_unique_brands": total_unique_brands,
            "total_mentions": len(mentions.data),
            "top_brands": [{"brand": brand, "mentions": count} for brand, count in top_brands],
            "sentiment_distribution": sentiment_counts,
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching brand statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/citation-stats")
async def get_citation_statistics():
    """
    Get statistics about extracted citations
    
    Returns:
        - Total citations
        - Domain category distribution
        - Top domains
    """
    try:
        supabase = citation_pipeline.supabase
        
        # Get all citations
        citations = supabase.table("citations").select(
            "citation_domain, domain_category, is_alive"
        ).execute()
        
        if not citations.data:
            return {
                "total_citations": 0,
                "domain_categories": {},
                "top_domains": [],
            }
        
        # Calculate statistics
        domain_counts = {}
        category_counts = {}
        alive_count = 0
        
        for citation in citations.data:
            domain = citation.get("citation_domain", "unknown")
            category = citation.get("domain_category", "Unknown")
            is_alive = citation.get("is_alive", False)
            
            # Count domains
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            # Count categories
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Count alive links
            if is_alive:
                alive_count += 1
        
        # Get top 20 domains
        top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return {
            "total_citations": len(citations.data),
            "alive_citations": alive_count,
            "domain_categories": category_counts,
            "top_domains": [{"domain": domain, "count": count} for domain, count in top_domains],
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching citation statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
