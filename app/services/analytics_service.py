from typing import Dict, Any, List, Optional
import pandas as pd
from app.utils.db import db_manager
from app.core.logger import logger


class AnalyticsCache:
    """In-memory cache for analytics data"""
    _instance = None
    _data_cache: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AnalyticsCache, cls).__new__(cls)
        return cls._instance
    
    def set(self, key: str, value: Any):
        self._data_cache[key] = value
    
    def get(self, key: str) -> Optional[Any]:
        return self._data_cache.get(key)
    
    def has(self, key: str) -> bool:
        return key in self._data_cache
    
    def clear(self):
        self._data_cache.clear()


# Global cache instance
analytics_cache = AnalyticsCache()


async def load_all_data() -> Dict[str, pd.DataFrame]:
    """
    Load all data from both tables into dataframes
    Returns dict with 'product_analysis' and 'normalized_score' dataframes
    """
    # Check if data already loaded
    if analytics_cache.has('dataframes'):
        logger.info("✅ Using cached dataframes")
        return analytics_cache.get('dataframes')
    
    logger.info("📊 Loading all data from Supabase...")
    
    try:
        # Fetch product_analysis data
        product_data = await db_manager.fetch_table_as_dataframe(
            "product_analysis",
            batch_size=1000  # Adjust based on your table size
        )
        logger.info(f"✅ Loaded product_analysis: {len(product_data)} rows")
        
        if len(product_data) > 0:
            logger.info(f"📋 Product columns: {list(product_data.columns)}")
        else:
            logger.error("❌ product_analysis table is EMPTY!")
        
        # Fetch normalized_score data
        score_data = await db_manager.fetch_table_as_dataframe(
            "normalized_score",
            batch_size=1000
        )
        logger.info(f"✅ Loaded normalized_score: {len(score_data)} rows")
        
        if len(score_data) > 0:
            logger.info(f"📋 Score columns: {list(score_data.columns)}")
        else:
            logger.error("❌ normalized_score table is EMPTY!")
        
        dataframes = {
            'product_analysis': product_data,
            'normalized_score': score_data
        }
        
        # Cache the dataframes
        analytics_cache.set('dataframes', dataframes)
        
        return dataframes
    
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise


async def get_unique_marketplaces() -> List[str]:
    """Get unique marketplaces from cached data"""
    
    # Check cache first
    if analytics_cache.has('unique_marketplaces'):
        return analytics_cache.get('unique_marketplaces')
    
    try:
        # Load data
        dataframes = await load_all_data()
        df = dataframes['product_analysis']
        
        # Validate DataFrame
        if df.empty:
            logger.error("❌ product_analysis DataFrame is empty")
            return []
        
        # Validate 'marketplaces' column exists
        if 'marketplaces' not in df.columns:
            logger.error(
                f"❌ 'marketplaces' column not found in product_analysis table. "
                f"Available columns: {list(df.columns)}"
            )
            return []
        
        # Get unique marketplaces (case-insensitive deduplication)
        marketplaces_dict = {}
        
        for marketplace in df['marketplaces'].dropna().unique():
            marketplace_clean = str(marketplace).strip()
            
            # Skip empty
            if not marketplace_clean:
                continue
            
            marketplace_lower = marketplace_clean.lower()
            
            if marketplace_lower not in marketplaces_dict:
                marketplaces_dict[marketplace_lower] = marketplace_clean
            else:
                # Prefer capitalized version
                existing = marketplaces_dict[marketplace_lower]
                if marketplace_clean[0].isupper() and not existing[0].isupper():
                    marketplaces_dict[marketplace_lower] = marketplace_clean
        
        marketplaces = list(marketplaces_dict.values())
        
        # Sort: amazon first, then alphabetical
        def sort_key(x):
            if x.lower() == "amazon":
                return (0, "")
            return (1, x.lower())
        
        marketplaces.sort(key=sort_key)
        
        # Cache the result
        analytics_cache.set('unique_marketplaces', marketplaces)
        
        logger.info(f"✅ Found {len(marketplaces)} unique marketplaces")
        return marketplaces
        
    except Exception as e:
        logger.error(f"❌ Error in get_unique_marketplaces: {e}")
        return []


async def calculate_marketplace_kpis(marketplace: str) -> Dict[str, Any]:
    """
    Calculate KPI metrics for a specific marketplace
    
    Returns:
        Dict with visibility_score, market_leadership, avg_ranking, opportunity_gap
    """
    
    # Check if already calculated for this marketplace
    cache_key = f'kpis_{marketplace.lower()}'
    if analytics_cache.has(cache_key):
        logger.info(f"✅ Using cached KPIs for {marketplace}")
        return analytics_cache.get(cache_key)
    
    logger.info(f"📊 Calculating KPIs for {marketplace}...")
    
    try:
        dataframes = await load_all_data()
        score_df = dataframes['normalized_score']
        
        # Validate data
        if score_df.empty:
            logger.error("❌ normalized_score DataFrame is empty")
            return {
                "error": "No data available",
                "visibility_score": {"percentage": 0, "present_in": 0, "total_categories": 0},
                "market_leadership": {"percentage": 0, "rank_1_count": 0, "total_categories": 0},
                "avg_ranking": {"value": 0, "rating": "no data"},
                "opportunity_gap": {"percentage": 0, "categories_not_rank1": 0, "total_categories": 0}
            }
        
        # Filter for this marketplace (case-insensitive)
        marketplace_df = score_df[
            score_df['marketplaces'].str.lower() == marketplace.lower()
        ].copy()
        
        # Total unique categories
        total_categories = score_df['product_category'].nunique()
        
        # 1. Visibility Score
        categories_present = len(marketplace_df)
        visibility_percentage = (categories_present / total_categories) * 100 if total_categories > 0 else 0
        
        # 2. Market Leadership (Rank #1 positions)
        rank_1_count = len(marketplace_df[marketplace_df['rank'] == 1])
        leadership_percentage = (rank_1_count / total_categories) * 100 if total_categories > 0 else 0
        
        # 3. Average Ranking
        avg_rank = marketplace_df['rank'].mean() if len(marketplace_df) > 0 else 0
        
        # 4. Opportunity Gap
        opportunity_gap_percentage = ((total_categories - rank_1_count) / total_categories) * 100 if total_categories > 0 else 0
        non_rank1_categories = total_categories - rank_1_count
        
        kpis = {
            "visibility_score": {
                "percentage": round(visibility_percentage, 2),
                "present_in": categories_present,
                "total_categories": total_categories
            },
            "market_leadership": {
                "percentage": round(leadership_percentage, 2),
                "rank_1_count": rank_1_count,
                "total_categories": total_categories
            },
            "avg_ranking": {
                "value": round(avg_rank, 2),
                "rating": "excellent" if avg_rank <= 2 else "good" if avg_rank <= 3 else "needs improvement"
            },
            "opportunity_gap": {
                "percentage": round(opportunity_gap_percentage, 2),
                "categories_not_rank1": non_rank1_categories,
                "total_categories": total_categories
            }
        }
        
        # Cache the result
        analytics_cache.set(cache_key, kpis)
        
        logger.info(f"✅ KPIs calculated for {marketplace}")
        return kpis
        
    except Exception as e:
        logger.error(f"❌ Error calculating KPIs for {marketplace}: {e}")
        return {
            "error": str(e),
            "visibility_score": {"percentage": 0, "present_in": 0, "total_categories": 0},
            "market_leadership": {"percentage": 0, "rank_1_count": 0, "total_categories": 0},
            "avg_ranking": {"value": 0, "rating": "error"},
            "opportunity_gap": {"percentage": 0, "categories_not_rank1": 0, "total_categories": 0}
        }


async def get_marketplace_heatmap(marketplace: str, limit: int = 5) -> Dict[str, Any]:
    """
    Get heatmap data showing top 5 categories and top 5 marketplaces for each
    Highlights competitors who are rank #1 when selected marketplace is not
    
    Args:
        marketplace: Selected marketplace name
        limit: Number of categories to show (default 5)
    
    Returns:
        Dict with heatmap data
    """
    
    cache_key = f'heatmap_{marketplace.lower()}_{limit}'
    if analytics_cache.has(cache_key):
        logger.info(f"✅ Using cached heatmap for {marketplace}")
        return analytics_cache.get(cache_key)
    
    logger.info(f"📊 Generating heatmap for {marketplace}...")
    
    dataframes = await load_all_data()
    score_df = dataframes['normalized_score']
    
    # Get first N categories (alphabetically)
    categories = sorted(score_df['product_category'].unique())[:limit]
    
    heatmap_data = []
    
    for category in categories:
        # Get top 5 marketplaces for this category
        category_data = score_df[score_df['product_category'] == category]\
            .sort_values('rank')\
            .head(5)
        
        row = {
            "category": category,
            "marketplaces": []
        }
        
        selected_marketplace_rank = None
        selected_marketplace_found = False
        
        for idx, row_data in category_data.iterrows():
            marketplace_name = row_data['marketplaces']
            rank = row_data['rank']
            
            marketplace_info = {
                "name": marketplace_name,
                "rank": rank,
                "score_norm": round(row_data['score_norm'], 4),
                "highlight": False
            }
            
            # Track selected marketplace rank (case-insensitive comparison)
            if str(marketplace_name).strip().lower() == marketplace.strip().lower():
                selected_marketplace_rank = rank
                selected_marketplace_found = True
            
            row["marketplaces"].append(marketplace_info)
        
        # Highlight rank #1 ONLY if:
        # 1. Selected marketplace exists in this category
        # 2. Selected marketplace is NOT rank #1
        if selected_marketplace_found and selected_marketplace_rank and selected_marketplace_rank > 1:
            for mp in row["marketplaces"]:
                if mp["rank"] == 1:
                    mp["highlight"] = True
                    logger.debug(f"Highlighting {mp['name']} (rank #1) in category {category} because {marketplace} is rank #{selected_marketplace_rank}")
                    break
        
        row["selected_marketplace_rank"] = selected_marketplace_rank
        heatmap_data.append(row)
    
    result = {
        "categories": heatmap_data,
        "selected_marketplace": marketplace
    }
    
    # Cache it
    analytics_cache.set(cache_key, result)
    
    logger.info(f"✅ Heatmap generated for {marketplace} with {len(heatmap_data)} categories")
    return result



async def get_marketplace_analytics(marketplace: str) -> Dict[str, Any]:
    """
    Get complete analytics for a marketplace
    Combines KPIs and heatmap data
    """
    
    cache_key = f'analytics_{marketplace.lower()}'
    if analytics_cache.has(cache_key):
        logger.info(f"✅ Using cached analytics for {marketplace}")
        return analytics_cache.get(cache_key)
    
    logger.info(f"📊 Generating complete analytics for {marketplace}...")
    
    # Calculate all metrics
    kpis = await calculate_marketplace_kpis(marketplace)
    heatmap = await get_marketplace_heatmap(marketplace, limit=5)
    
    analytics = {
        "marketplace": marketplace,
        "kpis": kpis,
        "heatmap": heatmap
    }
    
    # Cache complete analytics
    analytics_cache.set(cache_key, analytics)
    
    logger.info(f"✅ Complete analytics ready for {marketplace}")
    return analytics


async def get_detailed_heatmap(marketplace: str, show_all: bool = False) -> Dict[str, Any]:
    """
    Get detailed heatmap data showing all categories (or first 20) and top 10 marketplaces
    
    Args:
        marketplace: Selected marketplace name
        show_all: If True, show all categories; if False, show first 20
    
    Returns:
        Dict with detailed heatmap data
    """
    
    cache_key = f'detailed_heatmap_{marketplace.lower()}_{show_all}'
    if analytics_cache.has(cache_key):
        logger.info(f"✅ Using cached detailed heatmap for {marketplace}")
        return analytics_cache.get(cache_key)
    
    logger.info(f"📊 Generating detailed heatmap for {marketplace}...")
    
    dataframes = await load_all_data()
    score_df = dataframes['normalized_score']
    
    # Get categories
    all_categories = sorted(score_df['product_category'].unique())
    categories = all_categories if show_all else all_categories[:20]
    
    heatmap_data = []
    
    for category in categories:
        # Get top 10 marketplaces for this category
        category_data = score_df[score_df['product_category'] == category]\
            .sort_values('rank')\
            .head(10)
        
        row = {
            "category": category,
            "marketplaces": []
        }
        
        selected_marketplace_rank = None
        rank_1_marketplace = None
        
        for idx, row_data in category_data.iterrows():
            marketplace_name = row_data['marketplaces']
            rank = row_data['rank']
            
            marketplace_info = {
                "name": marketplace_name,
                "rank": rank,
                "score_norm": round(row_data['score_norm'], 4),
                "is_selected": False,
                "is_rank_1": False
            }
            
            # Track selected marketplace
            if str(marketplace_name).strip().lower() == marketplace.strip().lower():
                selected_marketplace_rank = rank
                marketplace_info["is_selected"] = True
            
            # Track rank #1
            if rank == 1:
                rank_1_marketplace = marketplace_name
                marketplace_info["is_rank_1"] = True
            
            row["marketplaces"].append(marketplace_info)
        
        row["selected_marketplace_rank"] = selected_marketplace_rank
        row["rank_1_marketplace"] = rank_1_marketplace
        heatmap_data.append(row)
    
    result = {
        "categories": heatmap_data,
        "selected_marketplace": marketplace,
        "total_categories": len(all_categories),
        "showing_categories": len(categories),
        "show_all": show_all
    }
    
    # Cache it
    analytics_cache.set(cache_key, result)
    
    logger.info(f"✅ Detailed heatmap generated for {marketplace}")
    return result


async def get_category_analytics(category: str, marketplace: str) -> Dict[str, Any]:
    """
    Get analytics for a specific product category
    All product counts are consistent and based on unique products in category
    Combines case-insensitive duplicate marketplaces (e.g., 'flipkart' and 'Flipkart')
    
    Args:
        category: Product category name
        marketplace: Selected marketplace name
    
    Returns:
        Dict with category analytics including KPIs, products, and marketplace rankings
    """
    
    cache_key = f'category_analytics_{category.lower()}_{marketplace.lower()}'
    if analytics_cache.has(cache_key):
        logger.info(f"✅ Using cached category analytics for {category} - {marketplace}")
        return analytics_cache.get(cache_key)
    
    logger.info(f"📊 Generating category analytics for {category} - {marketplace}...")
    
    dataframes = await load_all_data()
    product_df = dataframes['product_analysis']
    score_df = dataframes['normalized_score']
    
    # Filter data for this category
    category_products = product_df[product_df['product_category'] == category].copy()
    category_scores = score_df[score_df['product_category'] == category].copy()
    
    # Get ALL unique products in this category (baseline for all calculations)
    all_unique_products = category_products['product_name'].dropna().unique()
    total_category_products = int(len(all_unique_products))
    
    # Filter products where selected marketplace appears (case-insensitive)
    marketplace_products = category_products[
        category_products['marketplaces'].str.lower() == marketplace.lower()
    ].copy()
    
    # Get unique product names where selected marketplace appears
    marketplace_unique_products = marketplace_products['product_name'].dropna().unique()
    products = [str(p) for p in marketplace_unique_products]
    selected_marketplace_product_count = int(len(marketplace_unique_products))
    
    # KPI 2: Total unique marketplaces in category
    total_marketplaces = int(len(category_scores))
    
    # KPI 3: Top marketplace in category
    top_marketplace = category_scores[category_scores['rank'] == 1]['marketplaces'].values
    top_marketplace = str(top_marketplace[0]) if len(top_marketplace) > 0 else "N/A"
    
    # KPI 4: Selected marketplace rank
    selected_rank_data = category_scores[
        category_scores['marketplaces'].str.lower() == marketplace.lower()
    ]
    selected_rank = int(selected_rank_data['rank'].values[0]) if len(selected_rank_data) > 0 else None
    
    # Get marketplace rankings with product coverage
    # Combine case-insensitive duplicates
    marketplace_dict = {}
    
    for idx, row in category_scores.iterrows():
        marketplace_name = str(row['marketplaces'])
        marketplace_lower = marketplace_name.lower()
        rank = int(row['rank'])
        
        # Count UNIQUE products where this marketplace appears (case-insensitive)
        marketplace_prods = category_products[
            category_products['marketplaces'].str.lower() == marketplace_lower
        ]
        unique_marketplace_prods = marketplace_prods['product_name'].dropna().unique()
        product_count = int(len(unique_marketplace_prods))
        
        # If marketplace already exists (case-insensitive), combine data
        if marketplace_lower in marketplace_dict:
            # Keep the better rank and add product counts
            existing = marketplace_dict[marketplace_lower]
            marketplace_dict[marketplace_lower] = {
                "rank": min(existing["rank"], rank),  # Keep best rank
                "marketplace": existing["marketplace"] if existing["marketplace"][0].isupper() else marketplace_name,  # Prefer capitalized
                "product_coverage": max(existing["product_coverage"], product_count),  # Keep max count
                "total_products": total_category_products,
                "score_norm": max(existing["score_norm"], float(row['score_norm']))  # Keep best score
            }
        else:
            # Only include marketplaces with at least 1 product
            if product_count > 0:
                marketplace_dict[marketplace_lower] = {
                    "rank": rank,
                    "marketplace": marketplace_name,
                    "product_coverage": product_count,
                    "total_products": total_category_products,
                    "score_norm": float(round(row['score_norm'], 4))
                }
    
    # Convert to list and sort by rank
    marketplace_rankings = sorted(marketplace_dict.values(), key=lambda x: x["rank"])
    
    analytics = {
        "category": str(category),
        "selected_marketplace": str(marketplace),
        "kpis": {
            "total_products": selected_marketplace_product_count,
            "total_marketplaces": total_marketplaces,
            "top_marketplace": top_marketplace,
            "selected_marketplace_rank": selected_rank
        },
        "products": products,
        "marketplace_rankings": marketplace_rankings,
        "total_category_products": total_category_products
    }
    
    # Cache it
    analytics_cache.set(cache_key, analytics)
    
    logger.info(f"✅ Category analytics generated for {category} - {marketplace}")
    logger.info(f"   Total products in category: {total_category_products}")
    logger.info(f"   Products with {marketplace}: {selected_marketplace_product_count}")
    logger.info(f"   Unique marketplaces after deduplication: {len(marketplace_rankings)}")
    
    return analytics






async def get_all_categories() -> List[str]:
    """Get list of all product categories"""
    
    if analytics_cache.has('all_categories'):
        return analytics_cache.get('all_categories')
    
    dataframes = await load_all_data()
    score_df = dataframes['normalized_score']
    
    categories = sorted(score_df['product_category'].unique().tolist())
    
    analytics_cache.set('all_categories', categories)
    
    return categories


async def get_products_by_category(category: str) -> List[str]:
    """
    Get list of all unique products in a category
    
    Args:
        category: Product category name
    
    Returns:
        List of unique product names, sorted alphabetically
    """
    
    cache_key = f'products_by_category_{category.lower()}'
    if analytics_cache.has(cache_key):
        return analytics_cache.get(cache_key)
    
    dataframes = await load_all_data()
    product_df = dataframes['product_analysis']
    
    # Filter for category and get unique products
    category_products = product_df[product_df['product_category'] == category]
    products = sorted(category_products['product_name'].dropna().unique().tolist())
    
    # Cache it
    analytics_cache.set(cache_key, products)
    
    logger.info(f"✅ Found {len(products)} products in category '{category}'")
    return products


async def get_product_analytics(category: str, product_name: str) -> Dict[str, Any]:
    """
    Get analytics for a specific product
    Shows marketplace rankings for this product with podium data
    
    Args:
        category: Product category name
        product_name: Product name
    
    Returns:
        Dict with product analytics including marketplace podium
    """
    
    cache_key = f'product_analytics_{category.lower()}_{product_name.lower()}'
    if analytics_cache.has(cache_key):
        logger.info(f"✅ Using cached product analytics for {product_name}")
        return analytics_cache.get(cache_key)
    
    logger.info(f"📊 Generating product analytics for '{product_name}' in '{category}'...")
    
    dataframes = await load_all_data()
    product_df = dataframes['product_analysis']
    
    # Filter for this specific product
    product_data = product_df[
        (product_df['product_category'] == category) &
        (product_df['product_name'] == product_name)
    ].copy()
    
    if len(product_data) == 0:
        logger.warning(f"⚠️ No data found for product '{product_name}'")
        return {
            "category": category,
            "product_name": product_name,
            "marketplace_podium": [],
            "total_marketplaces": 0
        }
    
    # Get marketplace rankings for this product
    marketplace_data = []
    
    for idx, row in product_data.iterrows():
        marketplace_name = str(row['marketplaces'])
        rank = int(row['rank']) if pd.notna(row['rank']) else 999
        
        marketplace_info = {
            "marketplace": marketplace_name,
            "rank": rank,
            "price": float(row['price']) if pd.notna(row['price']) and row['price'] != -1 else None,
            "currency": str(row['price_currency']) if pd.notna(row['price_currency']) else None,
            "delivery_fee": float(row['delivery_fee']) if pd.notna(row['delivery_fee']) and row['delivery_fee'] != -1 else None,
            "delivery_days": int(row['delivery_days']) if pd.notna(row['delivery_days']) and row['delivery_days'] != -1 else None,
        }
        
        marketplace_data.append(marketplace_info)
    
    # Sort by rank
    marketplace_data.sort(key=lambda x: x['rank'])
    
    # Deduplicate case-insensitive
    marketplace_dict = {}
    for mp in marketplace_data:
        mp_lower = mp['marketplace'].lower()
        if mp_lower not in marketplace_dict:
            marketplace_dict[mp_lower] = mp
        else:
            # Keep better rank
            if mp['rank'] < marketplace_dict[mp_lower]['rank']:
                marketplace_dict[mp_lower] = mp
    
    marketplace_podium = list(marketplace_dict.values())
    marketplace_podium.sort(key=lambda x: x['rank'])
    
    analytics = {
        "category": str(category),
        "product_name": str(product_name),
        "marketplace_podium": marketplace_podium,
        "total_marketplaces": len(marketplace_podium)
    }
    
    # Cache it
    analytics_cache.set(cache_key, analytics)
    
    logger.info(f"✅ Product analytics generated for '{product_name}' ({len(marketplace_podium)} marketplaces)")
    return analytics


async def get_no_rank_analysis(marketplace: str, limit: int = 7) -> Dict[str, Any]:
    """
    Get top categories where marketplace has no rank (missing products)
    """
    
    cache_key = f'no_rank_analysis_{marketplace.lower()}_{limit}'
    if analytics_cache.has(cache_key):
        logger.info(f"✅ Using cached no-rank analysis for {marketplace}")
        return analytics_cache.get(cache_key)
    
    logger.info(f"📊 Generating no-rank analysis for {marketplace}...")
    
    try:
        dataframes = await load_all_data()
        product_df = dataframes['product_analysis']
        
        # Validate DataFrame
        if product_df.empty:
            logger.error("❌ product_analysis DataFrame is empty")
            return {
                "marketplace": marketplace,
                "top_no_rank_categories": [],
                "total_categories_analyzed": 0,
                "error": "No data available in product_analysis table"
            }
        
        # Validate required columns
        required_columns = ['product_category', 'marketplaces', 'product_name']
        missing_columns = [col for col in required_columns if col not in product_df.columns]
        
        if missing_columns:
            logger.error(
                f"❌ Missing required columns in product_analysis: {missing_columns}. "
                f"Available: {list(product_df.columns)}"
            )
            return {
                "marketplace": marketplace,
                "top_no_rank_categories": [],
                "total_categories_analyzed": 0,
                "error": f"Missing columns: {missing_columns}"
            }
        
        # Get all unique categories
        all_categories = product_df['product_category'].unique()
        
        no_rank_data = []
        
        for category in all_categories:
            # Get all products in this category
            category_products = product_df[product_df['product_category'] == category]
            total_products = len(category_products['product_name'].dropna().unique())
            
            # Get products where marketplace appears
            marketplace_products = category_products[
                category_products['marketplaces'].str.lower() == marketplace.lower()
            ]
            products_with_rank = len(marketplace_products['product_name'].dropna().unique())
            
            # Calculate missing products
            missing_products = total_products - products_with_rank
            missing_percentage = (missing_products / total_products * 100) if total_products > 0 else 0
            
            if missing_products > 0:
                no_rank_data.append({
                    "category": str(category),
                    "total_products": int(total_products),
                    "products_with_rank": int(products_with_rank),
                    "missing_products": int(missing_products),
                    "missing_percentage": round(missing_percentage, 2)
                })
        
        # Sort by missing products count (descending)
        no_rank_data.sort(key=lambda x: x['missing_products'], reverse=True)
        
        # Get top N
        top_categories = no_rank_data[:limit]
        
        result = {
            "marketplace": marketplace,
            "top_no_rank_categories": top_categories,
            "total_categories_analyzed": len(all_categories)
        }
        
        # Cache it
        analytics_cache.set(cache_key, result)
        
        logger.info(f"✅ No-rank analysis generated for {marketplace}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in get_no_rank_analysis: {e}")
        return {
            "marketplace": marketplace,
            "top_no_rank_categories": [],
            "total_categories_analyzed": 0,
            "error": str(e)
        }

async def get_low_rank_analysis(marketplace: str, limit: int = 7, low_rank_threshold: int = 5) -> Dict[str, Any]:
    """
    Get top categories where marketplace has low ranks (rank > threshold)
    
    Args:
        marketplace: Selected marketplace name
        limit: Number of top categories to return
        low_rank_threshold: Rank threshold (default: ranks > 5 considered low)
    
    Returns:
        Dict with low-rank analysis data
    """
    
    cache_key = f'low_rank_analysis_{marketplace.lower()}_{limit}_{low_rank_threshold}'
    if analytics_cache.has(cache_key):
        logger.info(f"✅ Using cached low-rank analysis for {marketplace}")
        return analytics_cache.get(cache_key)
    
    logger.info(f"📊 Generating low-rank analysis for {marketplace}...")
    
    dataframes = await load_all_data()
    product_df = dataframes['product_analysis']
    
    # Get all unique categories
    all_categories = product_df['product_category'].unique()
    
    low_rank_data = []
    
    for category in all_categories:
        # Get products where marketplace appears in this category
        marketplace_products = product_df[
            (product_df['product_category'] == category) &
            (product_df['marketplaces'].str.lower() == marketplace.lower())
        ].copy()
        
        if len(marketplace_products) == 0:
            continue
        
        # Count products with low ranks (rank > threshold)
        low_rank_products = marketplace_products[
            marketplace_products['rank'] > low_rank_threshold
        ]
        
        total_products = len(marketplace_products['product_name'].dropna().unique())
        low_rank_count = len(low_rank_products['product_name'].dropna().unique())
        
        # Calculate average rank
        avg_rank = marketplace_products['rank'].mean()
        
        if low_rank_count > 0:
            low_rank_percentage = (low_rank_count / total_products * 100) if total_products > 0 else 0
            
            low_rank_data.append({
                "category": str(category),
                "total_products": int(total_products),
                "low_rank_count": int(low_rank_count),
                "low_rank_percentage": round(low_rank_percentage, 2),
                "avg_rank": round(avg_rank, 2)
            })
    
    # Sort by low rank count (descending)
    low_rank_data.sort(key=lambda x: x['low_rank_count'], reverse=True)
    
    # Get top N
    top_categories = low_rank_data[:limit]
    
    result = {
        "marketplace": marketplace,
        "top_low_rank_categories": top_categories,
        "total_categories_analyzed": len(all_categories),
        "rank_threshold": low_rank_threshold
    }
    
    # Cache it
    analytics_cache.set(cache_key, result)
    
    logger.info(f"✅ Low-rank analysis generated for {marketplace}")
    return result


async def get_competitor_threat_analysis(marketplace: str, limit: int = 5) -> Dict[str, Any]:
    """
    Get top competitors analysis showing which marketplaces dominate most categories
    
    Args:
        marketplace: Selected marketplace name
        limit: Number of top competitors to return
    
    Returns:
        Dict with competitor threat analysis data
    """
    
    cache_key = f'competitor_threat_{marketplace.lower()}_{limit}'
    if analytics_cache.has(cache_key):
        logger.info(f"✅ Using cached competitor threat analysis for {marketplace}")
        return analytics_cache.get(cache_key)
    
    logger.info(f"📊 Generating competitor threat analysis for {marketplace}...")
    
    dataframes = await load_all_data()
    score_df = dataframes['normalized_score']
    
    # Get all categories
    all_categories = score_df['product_category'].unique()
    
    # Count rank #1 positions for each marketplace
    competitor_data = {}
    
    for category in all_categories:
        category_data = score_df[score_df['product_category'] == category]
        
        # Get rank #1 marketplace in this category
        rank_1_data = category_data[category_data['rank'] == 1]
        
        if len(rank_1_data) > 0:
            rank_1_marketplace = str(rank_1_data.iloc[0]['marketplaces'])
            rank_1_lower = rank_1_marketplace.lower()
            
            # Skip if it's the selected marketplace
            if rank_1_lower == marketplace.lower():
                continue
            
            # Initialize or update competitor count
            if rank_1_lower not in competitor_data:
                competitor_data[rank_1_lower] = {
                    "marketplace": rank_1_marketplace,
                    "categories_dominated": 0,
                    "categories": []
                }
            
            competitor_data[rank_1_lower]["categories_dominated"] += 1
            competitor_data[rank_1_lower]["categories"].append(str(category))
    
    # Convert to list and sort by categories dominated
    competitors = list(competitor_data.values())
    competitors.sort(key=lambda x: x["categories_dominated"], reverse=True)
    
    # Get top N competitors
    top_competitors = competitors[:limit]
    
    # Calculate percentages
    total_categories = len(all_categories)
    for comp in top_competitors:
        comp["dominance_percentage"] = round(
            (comp["categories_dominated"] / total_categories) * 100, 2
        )
    
    result = {
        "marketplace": marketplace,
        "top_competitors": top_competitors,
        "total_categories": int(total_categories),
        "analysis_date": pd.Timestamp.now().isoformat()
    }
    
    # Cache it
    analytics_cache.set(cache_key, result)
    
    logger.info(f"✅ Competitor threat analysis generated for {marketplace}")
    logger.info(f"   Top competitor: {top_competitors[0]['marketplace'] if top_competitors else 'None'}")
    
    return result
