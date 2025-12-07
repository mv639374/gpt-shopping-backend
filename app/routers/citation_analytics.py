from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from datetime import datetime
from app.utils.db import get_supabase_client
from app.core.logger import logger
from collections import defaultdict

router = APIRouter(prefix="/citations/analytics", tags=["Citation Analytics"])


def chunk_list(lst: List, chunk_size: int = 100):
    """Split a list into chunks"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


@router.get("/metrics")
async def get_citation_analytics(
    brand_id: Optional[str] = Query(None),
    regions: Optional[List[str]] = Query(None),
    platforms: Optional[List[str]] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    min_citations: int = Query(1, ge=1),
):
    """
    Get citation analytics metrics for a specific child brand
    """
    try:
        supabase = get_supabase_client()
        
        if not brand_id:
            return get_empty_metrics([])
        
        logger.info(f"🔍 Fetching citation analytics for brand: {brand_id}")
        
        # Step 1: Get responses with filters - but limit the data we fetch
        responses_query = supabase.table("responses-2").select("id")
        
        if regions and len(regions) > 0:
            responses_query = responses_query.in_("region", regions)
        if platforms and len(platforms) > 0:
            responses_query = responses_query.in_("platform", platforms)
        if start_date:
            responses_query = responses_query.gte("created_at", start_date)
        if end_date:
            responses_query = responses_query.lte("created_at", end_date)
        
        # Limit to reasonable number
        responses_query = responses_query.limit(1000)
        
        all_responses = responses_query.execute()
        response_ids = [r["id"] for r in all_responses.data] if all_responses.data else []
        
        if not response_ids:
            logger.warning("No responses found matching filters")
            return get_empty_metrics([brand_id])
        
        logger.info(f"📊 Found {len(response_ids)} responses matching filters")
        
        # Step 2: Get citations in batches to avoid JSON generation error
        all_citations = []
        for response_id_chunk in chunk_list(response_ids, 100):
            citations_response = supabase.table("citations").select("*").in_("prompt_response_id", response_id_chunk).execute()
            if citations_response.data:
                all_citations.extend(citations_response.data)
        
        logger.info(f"📊 Found {len(all_citations)} total citations")
        
        # Step 3: Get brand mentions in batches
        brand_mentions = []
        for response_id_chunk in chunk_list(response_ids, 100):
            mentions_response = supabase.table("response_brand_mentions").select("prompt_response_id, brand_id").eq("brand_id", brand_id).in_("prompt_response_id", response_id_chunk).execute()
            if mentions_response.data:
                brand_mentions.extend(mentions_response.data)
        
        brand_specific_response_ids = list(set([m["prompt_response_id"] for m in brand_mentions]))
        logger.info(f"📊 Found {len(brand_specific_response_ids)} responses with brand mentions")
        
        # Step 4: Filter citations for brand-specific responses
        brand_citations = [c for c in all_citations if c["prompt_response_id"] in brand_specific_response_ids]
        
        # Process metrics
        overall_metrics = process_citations_data(all_citations, len(response_ids))
        brand_metrics = process_citations_data(brand_citations, len(brand_specific_response_ids))
        
        logger.info(f"✅ Analytics complete: {overall_metrics['totalCitations']} overall citations, {brand_metrics['totalCitations']} brand-specific citations")
        
        return {
            "overall": overall_metrics,
            "brandSpecific": {
                **brand_metrics,
                "selectedChildBrandIds": [brand_id]
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching citation analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/source-citations")
async def get_source_citation_analytics(
    brand_id: Optional[str] = Query(None),
    regions: Optional[List[str]] = Query(None),
    platforms: Optional[List[str]] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """
    Get source citation analytics (citations that reference YOUR brand)
    """
    try:
        supabase = get_supabase_client()
        
        if not brand_id:
            return {
                "domainStats": [],
                "totalSourceCitations": 0,
                "totalDomains": 0,
                "totalBrands": 0,
                "selectedChildBrandIds": []
            }
        
        logger.info(f"🔍 Fetching source citations for brand: {brand_id}")
        
        # Get responses with filters - limited
        responses_query = supabase.table("responses-2").select("id")
        
        if regions and len(regions) > 0:
            responses_query = responses_query.in_("region", regions)
        if platforms and len(platforms) > 0:
            responses_query = responses_query.in_("platform", platforms)
        if start_date:
            responses_query = responses_query.gte("created_at", start_date)
        if end_date:
            responses_query = responses_query.lte("created_at", end_date)
        
        responses_query = responses_query.limit(1000)
        
        all_responses = responses_query.execute()
        response_ids = [r["id"] for r in all_responses.data] if all_responses.data else []
        
        if not response_ids:
            return {
                "domainStats": [],
                "totalSourceCitations": 0,
                "totalDomains": 0,
                "totalBrands": 0,
                "selectedChildBrandIds": [brand_id]
            }
        
        # Get brand mentions in batches
        brand_mentions = []
        for response_id_chunk in chunk_list(response_ids, 100):
            mentions_query = supabase.table("response_brand_mentions").select("*").eq("brand_id", brand_id).in_("prompt_response_id", response_id_chunk)
            mentions_response = mentions_query.execute()
            if mentions_response.data:
                brand_mentions.extend(mentions_response.data)
        
        # Process source citations
        domain_stats = defaultdict(lambda: {
            "citationCount": 0,
            "brandNames": set(),
            "urls": set(),
        })
        
        for mention in brand_mentions:
            source_urls = mention.get("source_citation_url", []) or []
            brand_name = mention.get("brand_name", "")
            
            for url in source_urls:
                if not url:
                    continue
                    
                # Extract domain from URL
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    domain = parsed.netloc.lower()
                    if domain.startswith("www."):
                        domain = domain[4:]
                except:
                    continue
                
                domain_stats[domain]["citationCount"] += 1
                domain_stats[domain]["brandNames"].add(brand_name)
                domain_stats[domain]["urls"].add(url)
        
        # Format domain stats
        formatted_domain_stats = []
        for domain, stats in domain_stats.items():
            formatted_domain_stats.append({
                "domain": domain,
                "citationCount": stats["citationCount"],
                "uniqueBrands": len(stats["brandNames"]),
                "brandNames": list(stats["brandNames"]),
                "urls": list(stats["urls"]),
                "urlDetails": [{"url": url, "citationCount": 1} for url in stats["urls"]],
                "platforms": [],
                "regions": [],
                "recentCitation": datetime.utcnow().isoformat(),
                "sentimentDistribution": {"positive": 0, "negative": 0, "neutral": 0},
                "avgPosition": 0,
                "domainCategory": None
            })
        
        formatted_domain_stats.sort(key=lambda x: x["citationCount"], reverse=True)
        
        total_source_citations = sum(s["citationCount"] for s in formatted_domain_stats)
        
        return {
            "domainStats": formatted_domain_stats,
            "totalSourceCitations": total_source_citations,
            "totalDomains": len(formatted_domain_stats),
            "totalBrands": 1,
            "selectedChildBrandIds": [brand_id]
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching source citations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def get_empty_metrics(brand_ids: List[str]):
    """Return empty metrics structure"""
    return {
        "overall": {
            "totalCitations": 0,
            "totalDomains": 0,
            "totalResponses": 0,
            "totalCitedPages": 0,
            "responsesWithCitations": 0,
            "domainCategoryDistribution": [],
            "topCitations": []
        },
        "brandSpecific": {
            "totalCitations": 0,
            "totalDomains": 0,
            "totalResponses": 0,
            "selectedChildBrandIds": brand_ids
        }
    }


def process_citations_data(citations: List[dict], total_responses: int) -> dict:
    """Helper function to process citation data into metrics"""
    
    # Domain metrics
    domain_metrics = defaultdict(lambda: {
        "citations": [],
        "uniqueResponses": set()
    })
    
    for citation in citations:
        domain = citation.get("citation_domain")
        if not domain:
            continue
        
        domain_metrics[domain]["citations"].append(citation)
        domain_metrics[domain]["uniqueResponses"].add(citation["prompt_response_id"])
    
    # Domain category distribution
    category_counts = defaultdict(int)
    for citation in citations:
        category = citation.get("domain_category") or "Unknown"
        category_counts[category] += 1
    
    total_citations = len(citations)
    domain_category_distribution = [
        {
            "category": category,
            "count": count,
            "percentage": (count / total_citations * 100) if total_citations > 0 else 0
        }
        for category, count in category_counts.items()
    ]
    domain_category_distribution.sort(key=lambda x: x["count"], reverse=True)
    
    # Top citation pages
    url_groups = defaultdict(lambda: {
        "citations": [],
        "count": 0,
        "mostRecent": None
    })
    
    for citation in citations:
        url = citation.get("citation_url")
        if not url:
            continue
        
        url_groups[url]["citations"].append(citation)
        url_groups[url]["count"] += 1
        
        if not url_groups[url]["mostRecent"] or citation.get("created_at", "") > url_groups[url]["mostRecent"].get("created_at", ""):
            url_groups[url]["mostRecent"] = citation
    
    top_citation_pages = []
    for url, data in url_groups.items():
        most_recent = data["mostRecent"]
        if most_recent:
            top_citation_pages.append({
                "id": most_recent["id"],
                "url": url,
                "title": most_recent.get("title"),
                "description": most_recent.get("description"),
                "domain": most_recent.get("citation_domain"),
                "citationType": most_recent.get("citation_type"),
                "domainCategory": most_recent.get("domain_category"),
                "utmSource": most_recent.get("utm_source"),
                "isAlive": most_recent.get("is_alive"),
                "createdAt": most_recent.get("created_at"),
                "text": most_recent.get("text"),
                "citationCount": data["count"]
            })
    
    top_citation_pages.sort(key=lambda x: x["citationCount"], reverse=True)
    
    # Responses with citations
    responses_with_citations = len(set(c["prompt_response_id"] for c in citations))
    
    return {
        "totalCitations": total_citations,
        "totalDomains": len(domain_metrics),
        "totalResponses": total_responses,
        "totalCitedPages": len(url_groups),
        "responsesWithCitations": responses_with_citations,
        "domainCategoryDistribution": domain_category_distribution,
        "topCitations": top_citation_pages[:20]  # Top 20
    }
