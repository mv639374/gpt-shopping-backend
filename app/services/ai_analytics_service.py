from typing import Dict, Any, List
import pandas as pd
from collections import Counter
from app.services.citation_service import CitationExtractor
from app.services.analytics_service import load_all_data, AnalyticsCache
from app.core.logger import logger
from textblob import TextBlob

analytics_cache = AnalyticsCache()


async def calculate_citation_metrics(marketplace: str) -> Dict[str, Any]:
    """
    Calculate citation performance metrics for a marketplace
    """
    
    cache_key = f'citation_metrics_{marketplace.lower()}'
    if analytics_cache.has(cache_key):
        logger.info(f"✅ Using cached citation metrics for {marketplace}")
        return analytics_cache.get(cache_key)
    
    logger.info(f"📊 Calculating citation metrics for {marketplace}...")
    
    dataframes = await load_all_data()
    product_df = dataframes['product_analysis']
    
    # Initialize metrics
    total_responses = 0
    responses_with_citations = 0
    marketplace_cited_count = 0
    marketplace_mentioned_count = 0
    
    all_citations = []
    citation_types = Counter()
    citation_sources = Counter()
    
    # Process each response
    for idx, row in product_df.iterrows():
        response = row.get('response', '')
        marketplace_name = str(row.get('marketplaces', '')).strip()
        
        if pd.isna(response) or not response:
            continue
        
        total_responses += 1
        
        # Extract citations
        citations = CitationExtractor.extract_citations_from_response(response)
        
        if citations:
            responses_with_citations += 1
        
        # Check if marketplace is mentioned (case-insensitive)
        marketplace_mentioned = marketplace.lower() in str(marketplace_name).lower()
        
        if marketplace_mentioned:
            marketplace_mentioned_count += 1
        
        # Process citations for this response
        for citation in citations:
            citation_type = CitationExtractor.classify_citation(citation['domain'], marketplace)
            citation_types[citation_type] += 1
            citation_sources[citation['domain']] += 1
            
            all_citations.append({
                'domain': citation['domain'],
                'type': citation_type,
                'url': citation['url'],
                'category': row.get('product_category', 'Unknown'),
                'marketplace_mentioned': marketplace_mentioned
            })
            
            # Count if this specific marketplace is cited
            if marketplace_mentioned:
                marketplace_cited_count += 1
    
    # Calculate metrics
    total_citations = len(all_citations)
    
    citation_share = (marketplace_cited_count / total_citations * 100) if total_citations > 0 else 0
    response_inclusion_rate = (marketplace_mentioned_count / total_responses * 100) if total_responses > 0 else 0
    citation_vs_no_citation = (marketplace_cited_count / marketplace_mentioned_count * 100) if marketplace_mentioned_count > 0 else 0
    
    # Get top citation sources (top 5)
    top_sources = [
        {
            'domain': domain,
            'count': count,
            'share': round(count / total_citations * 100, 2) if total_citations > 0 else 0
        }
        for domain, count in citation_sources.most_common(5)
    ]
    
    # Citation types breakdown
    citation_type_breakdown = {
        citation_type: {
            'count': count,
            'percentage': round(count / total_citations * 100, 2) if total_citations > 0 else 0
        }
        for citation_type, count in citation_types.items()
    }
    
    metrics = {
        'marketplace': marketplace,
        'citation_share': round(citation_share, 2),
        'citation_rank': None,  # Will be calculated in comparison with other marketplaces
        'response_inclusion_rate': round(response_inclusion_rate, 2),
        'total_citations': marketplace_cited_count,
        'total_responses': total_responses,
        'responses_with_citations': responses_with_citations,
        'citation_vs_no_citation': round(citation_vs_no_citation, 2),
        'top_citation_sources': top_sources,
        'citation_types': citation_type_breakdown,
        'trend': {
            'citation_share_change': 0,  # Placeholder for time-series analysis
            'rank_change': 0
        }
    }
    
    # Cache it
    analytics_cache.set(cache_key, metrics)
    
    logger.info(f"✅ Citation metrics calculated for {marketplace}")
    return metrics


async def calculate_prompt_intelligence(marketplace: str) -> Dict[str, Any]:
    """
    Calculate Share of Voice and prompt intelligence metrics
    """
    
    cache_key = f'prompt_intelligence_{marketplace.lower()}'
    if analytics_cache.has(cache_key):
        logger.info(f"✅ Using cached prompt intelligence for {marketplace}")
        return analytics_cache.get(cache_key)
    
    logger.info(f"📊 Calculating prompt intelligence for {marketplace}...")
    
    dataframes = await load_all_data()
    product_df = dataframes['product_analysis']
    
    # Count total unique prompts
    total_prompts = len(product_df['prompts'].dropna().unique())
    
    # Count mentions by marketplace
    marketplace_mentions = Counter()
    total_mentions = 0
    
    marketplace_prompts = []
    marketplace_ranks = []
    
    for idx, row in product_df.iterrows():
        marketplace_name = str(row.get('marketplaces', '')).strip()
        prompt = row.get('prompts', '')
        rank = row.get('rank', None)
        category = row.get('product_category', 'Unknown')
        
        if pd.isna(marketplace_name) or not marketplace_name:
            continue
        
        marketplace_lower = marketplace_name.lower()
        marketplace_mentions[marketplace_lower] += 1
        total_mentions += 1
        
        # Track data for selected marketplace
        if marketplace_lower == marketplace.lower():
            if not pd.isna(prompt):
                marketplace_prompts.append({
                    'prompt': prompt,
                    'category': category,
                    'rank': rank if not pd.isna(rank) else None
                })
            
            if not pd.isna(rank):
                marketplace_ranks.append(int(rank))
    
    # Calculate Share of Voice
    sov = (marketplace_mentions[marketplace.lower()] / total_mentions * 100) if total_mentions > 0 else 0
    
    # Calculate SoV rank
    sorted_marketplaces = sorted(marketplace_mentions.items(), key=lambda x: x[1], reverse=True)
    sov_rank = next((idx + 1 for idx, (mp, _) in enumerate(sorted_marketplaces) if mp == marketplace.lower()), None)
    
    # Calculate average rank
    avg_rank = sum(marketplace_ranks) / len(marketplace_ranks) if marketplace_ranks else 0
    
    # Find top trigger topics (categories with best ranks)
    category_ranks = {}
    for item in marketplace_prompts:
        category = item['category']
        rank = item['rank']
        
        if rank is not None:
            if category not in category_ranks:
                category_ranks[category] = []
            category_ranks[category].append(rank)
    
    # Calculate average rank per category
    category_avg_ranks = {
        cat: sum(ranks) / len(ranks)
        for cat, ranks in category_ranks.items()
    }
    
    # Get top 3 categories (best ranks)
    top_categories = sorted(category_avg_ranks.items(), key=lambda x: x[1])[:3]
    
    # Get weakness categories (worst ranks or missing)
    weakness_categories = sorted(category_avg_ranks.items(), key=lambda x: x[1], reverse=True)[:3]
    
    metrics = {
        'marketplace': marketplace,
        'share_of_voice': round(sov, 2),
        'sov_rank': sov_rank,
        'total_unique_prompts': total_prompts,
        'marketplace_prompt_count': len(marketplace_prompts),
        'average_rank': round(avg_rank, 2),
        'top_trigger_topics': [
            {'topic': cat, 'avg_rank': round(rank, 2)}
            for cat, rank in top_categories
        ],
        'weakness_topics': [
            {'topic': cat, 'avg_rank': round(rank, 2)}
            for cat, rank in weakness_categories
        ],
        'trend': {
            'sov_change': 0,  # Placeholder
            'rank_change': 0
        }
    }
    
    # Cache it
    analytics_cache.set(cache_key, metrics)
    
    logger.info(f"✅ Prompt intelligence calculated for {marketplace}")
    return metrics


async def prompt_grouping_by_category(marketplace: str) -> Dict[str, Any]:
    """
    Group prompts by product_category, return prompts, their avg rank, SoV, top competitors.
    """
    dataframes = await load_all_data()
    product_df = dataframes['product_analysis']

    category_prompts = {}

    for _, row in product_df.iterrows():
        category = str(row.get('product_category','')).strip()
        prompt = str(row.get('prompts','')).strip()
        mp = str(row.get('marketplaces','')).lower()
        rank = row.get('rank', None)
        if not prompt or not category: continue
        if category not in category_prompts: category_prompts[category] = {}
        stats = category_prompts[category].get(prompt, {'mentions':0,'our_ranks':[],'competitors':set()})
        if mp: stats['competitors'].add(mp)
        stats['mentions'] += 1
        if mp == marketplace.lower() and isinstance(rank, (int,float)):
            stats['our_ranks'].append(rank)
        category_prompts[category][prompt] = stats

    out = []
    for cat, prompts in category_prompts.items():
        for prm, d in prompts.items():
            avg_rank = (sum(d['our_ranks']) / len(d['our_ranks'])) if d['our_ranks'] else None
            out.append({
                'category': cat, 'prompt': prm, 'mentions': d['mentions'],
                'avg_rank': avg_rank,
                'competitors': sorted(list(d['competitors']))
            })
    return {'marketplace': marketplace, 'grouped_prompts': out}


async def fetch_prompt_detail(marketplace: str, prompt: str) -> Dict[str, Any]:
    """Detailed stats for single prompt"""
    dataframes = await load_all_data()
    df = dataframes['product_analysis']
    records = df[df['prompts'] == prompt]
    pod = []
    mentions = 0; my_rank = None; citations = []
    for _, row in records.iterrows():
        mp = str(row['marketplaces']).lower()
        rank = row['rank']; response = row['response']
        if mp == marketplace.lower(): my_rank = rank
        pod.append({"marketplace": mp, "rank": rank})
        cits = CitationExtractor.extract_citations_from_response(response)
        citations += [c for c in cits]
        mentions += 1
    return {
        'prompt': prompt, 'our_rank': my_rank, 'total_mentions': mentions,
        'podium': sorted(pod, key=lambda x: x['rank'] if x['rank'] is not None else 999)[:3],
        'citations': citations
    }


async def fetch_citation_source_detail(domain: str, limit: int=10) -> Dict[str, Any]:
    dataframes = await load_all_data()
    df = dataframes['product_analysis']
    found = []
    for _, row in df.iterrows():
        resp = row.get('response', '')
        cits = CitationExtractor.extract_citations_from_response(resp)
        for c in cits:
            if c['domain'] == domain:
                found.append({
                    'prompt': row.get('prompts'),
                    'category': row.get('product_category'),
                    'cit_url': c['url'],
                })
    return {'domain': domain, 'citations': found[:limit]}


async def sentiment_mini_dashboard(marketplace: str) -> Dict[str, Any]:
    """
    Sentiment analysis for all answers mentioning a marketplace.
    """
    dataframes = await load_all_data()
    df = dataframes['product_analysis']
    pos=neg=neu=total=0; by_cat={}; trend=[]
    for _, row in df.iterrows():
        mp = str(row.get('marketplaces','')).lower()
        resp = str(row.get('response',''))
        cat = str(row.get('product_category','')).strip()
        if mp != marketplace.lower() or not resp: continue
        sentiment = TextBlob(resp).sentiment.polarity
        label = 'positive' if sentiment > 0.1 else 'negative' if sentiment < -0.1 else 'neutral'
        by_cat.setdefault(cat, {'positive':0,'negative':0,'neutral':0,'total':0})
        by_cat[cat][label] += 1; by_cat[cat]['total'] += 1
        if label=='positive': pos+=1
        elif label=='negative': neg+=1
        else: neu += 1; total += 1

    return {
        'marketplace': marketplace,
        'score': round((pos-neg) / total * 100,2) if total else 0,
        'breakdown': {'positive': pos, 'negative': neg, 'neutral': neu, 'total': total},
        'by_category': by_cat,
        'example': "Uses TextBlob. Plug in better NLP model for higher accuracy." # demo only
    }