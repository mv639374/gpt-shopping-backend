from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from app.services.analytics_service import (
    get_unique_marketplaces,
    get_marketplace_analytics,
    load_all_data,
    get_detailed_heatmap,      # ← Added
    get_category_analytics,     # ← Added
    get_all_categories,
    get_products_by_category,
    get_product_analytics,
    get_no_rank_analysis,
    get_low_rank_analysis 
)
from app.services.ai_analytics_service import (
    prompt_grouping_by_category,
    fetch_prompt_detail,
    fetch_citation_source_detail,
    sentiment_mini_dashboard

)
from app.services.citation_service import CitationExtractor
from app.core.logger import logger

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/marketplaces", response_model=List[str])
async def get_marketplaces():
    """
    Get list of unique marketplaces
    Returns sorted list with 'amazon' first, then alphabetical
    """
    try:
        marketplaces = await get_unique_marketplaces()
        
        if not marketplaces:
            logger.warning("⚠️ No marketplaces found")
            return []
        
        return marketplaces
    
    except Exception as e:
        logger.error(f"❌ Error in get_marketplaces endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load-data")
async def load_data():
    """
    Force load all data from database into memory
    Called when user clicks 'Enter Dashboard'
    """
    try:
        logger.info("📥 Loading all data from database...")
        dataframes = await load_all_data()
        
        product_rows = len(dataframes['product_analysis'])
        score_rows = len(dataframes['normalized_score'])
        
        return {
            "status": "success",
            "message": "Data loaded successfully",
            "product_analysis_rows": product_rows,
            "normalized_score_rows": score_rows
        }
    
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/marketplace/{marketplace_name}")
async def get_marketplace_data(marketplace_name: str) -> Dict[str, Any]:
    """
    Get complete analytics data for a specific marketplace
    Includes KPIs and heatmap
    """
    try:
        logger.info(f"📊 Fetching analytics for '{marketplace_name}'")
        analytics = await get_marketplace_analytics(marketplace_name)
        return analytics
    
    except Exception as e:
        logger.error(f"❌ Error fetching data for '{marketplace_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/detailed-heatmap/{marketplace_name}")
async def get_detailed_heatmap_data(marketplace_name: str, show_all: bool = False):
    """
    Get detailed heatmap data showing 20 or all categories with top 10 marketplaces
    """
    try:
        logger.info(f"📊 Fetching detailed heatmap for '{marketplace_name}' (show_all={show_all})")
        heatmap = await get_detailed_heatmap(marketplace_name, show_all)
        return heatmap
    
    except Exception as e:
        logger.error(f"❌ Error fetching detailed heatmap: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/category/{category_name}/{marketplace_name}")
async def get_category_data(category_name: str, marketplace_name: str):
    """
    Get analytics for a specific product category and marketplace
    """
    try:
        logger.info(f"📊 Fetching category analytics for '{category_name}' - '{marketplace_name}'")
        analytics = await get_category_analytics(category_name, marketplace_name)
        return analytics
    
    except Exception as e:
        logger.error(f"❌ Error fetching category analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
async def get_categories_list():
    """
    Get list of all product categories
    """
    try:
        categories = await get_all_categories()
        return {"categories": categories}
    
    except Exception as e:
        logger.error(f"❌ Error fetching categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/products/{category_name}")
async def get_products_in_category(category_name: str):
    """
    Get list of all products in a category
    """
    try:
        # from app.services.analytics_service import get_products_by_category
        products = await get_products_by_category(category_name)
        return {"products": products}
    
    except Exception as e:
        logger.error(f"❌ Error fetching products for category '{category_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/product/{category_name}/{product_name}")
async def get_product_data(category_name: str, product_name: str):
    """
    Get analytics for a specific product
    """
    try:
        # from app.services.analytics_service import get_product_analytics
        analytics = await get_product_analytics(category_name, product_name)
        return analytics
    
    except Exception as e:
        logger.error(f"❌ Error fetching product analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/no-rank-analysis/{marketplace_name}")
async def get_no_rank_data(marketplace_name: str, limit: int = 7):
    """
    Get no-rank analysis for a marketplace
    Shows categories where marketplace is missing most
    """
    try:
        from app.services.analytics_service import get_no_rank_analysis
        analysis = await get_no_rank_analysis(marketplace_name, limit)
        return analysis
    
    except Exception as e:
        logger.error(f"❌ Error fetching no-rank analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/low-rank-analysis/{marketplace_name}")
async def get_low_rank_data(marketplace_name: str, limit: int = 7):
    """
    Get low-rank analysis for a marketplace
    Shows categories where marketplace has lowest ranks
    """
    try:
        from app.services.analytics_service import get_low_rank_analysis
        analysis = await get_low_rank_analysis(marketplace_name, limit)
        return analysis
    
    except Exception as e:
        logger.error(f"❌ Error fetching low-rank analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/competitor-threat/{marketplace_name}")
async def get_competitor_threat(marketplace_name: str, limit: int = 5):
    """
    Get competitor threat analysis for a marketplace
    Shows top competitors dominating most categories
    """
    try:
        from app.services.analytics_service import get_competitor_threat_analysis
        analysis = await get_competitor_threat_analysis(marketplace_name, limit)
        return analysis
    
    except Exception as e:
        logger.error(f"❌ Error fetching competitor threat analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/citation-metrics/{marketplace_name}")
async def get_citation_metrics(marketplace_name: str):
    """
    Get citation performance metrics for a marketplace
    """
    try:
        from app.services.ai_analytics_service import calculate_citation_metrics
        metrics = await calculate_citation_metrics(marketplace_name)
        return metrics
    
    except Exception as e:
        logger.error(f"❌ Error fetching citation metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prompt-intelligence/{marketplace_name}")
async def get_prompt_intelligence(marketplace_name: str):
    """
    Get prompt intelligence and Share of Voice metrics
    """
    try:
        from app.services.ai_analytics_service import calculate_prompt_intelligence
        metrics = await calculate_prompt_intelligence(marketplace_name)
        return metrics
    
    except Exception as e:
        logger.error(f"❌ Error fetching prompt intelligence: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prompts-by-category/{marketplace}")
async def get_prompt_grouping(marketplace: str):
    return await prompt_grouping_by_category(marketplace)

@router.get("/prompt-detail/{marketplace}")
async def get_prompt_detail(marketplace: str, prompt: str):
    return await fetch_prompt_detail(marketplace, prompt)

@router.get("/citation-source/{domain}")
async def get_citation_source_detail(domain: str):
    return await fetch_citation_source_detail(domain)

@router.get("/sentiment-dashboard/{marketplace}")
async def sentiment_dashboard(marketplace: str):
    return await sentiment_mini_dashboard(marketplace)


@router.get("/citation-detail/{marketplace_name}")
async def get_citation_detail(marketplace_name: str, limit: int = 10):
    """
    Returns all citation details for a marketplace:
    - Top cited pages, by freq and domain
    - All classified citations (Owned/Competition/Social/Earned)
    - Trends (stub, implement later)
    """
    from app.services.ai_analytics_service import calculate_citation_metrics, fetch_citation_source_detail

    # Main metrics
    metrics = await calculate_citation_metrics(marketplace_name)
    
    # Top cited pages across all responses (for now, get all, later paginate)
    dataframes = await load_all_data()
    product_df = dataframes['product_analysis']

    all_citations = []
    for _, row in product_df.iterrows():
        response = row.get("response", "")
        cits = CitationExtractor.extract_citations_from_response(response)
        pr_cat = row.get("product_category")
        for cit in cits:
            all_citations.append({
                "domain": cit['domain'],
                "url": cit['url'],
                "category": pr_cat
            })
    # Aggregate by URL
    from collections import Counter
    url_counts = Counter([c['url'] for c in all_citations])
    domain_counts = Counter([c['domain'] for c in all_citations])
    top_pages = [
        {"url": url, "count": count, "domain": CitationExtractor._extract_domain(url)}
        for url, count in url_counts.most_common(limit)
    ]
    top_domains = [
        {"domain": domain, "count": count}
        for domain, count in domain_counts.most_common(limit)
    ]
    metrics['top_cited_pages'] = top_pages
    metrics['top_cited_domains'] = top_domains

    return metrics


@router.get("/citation-source/{domain}")
async def get_citation_source_detail(domain: str, limit: int=10):
    from app.services.ai_analytics_service import fetch_citation_source_detail
    return await fetch_citation_source_detail(domain, limit)
