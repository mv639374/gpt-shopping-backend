from typing import Optional, Dict, Any, List
from supabase import create_client, Client
from app.core.config import settings
from app.core.logger import logger, log_data_loading
import pandas as pd
import os
from functools import lru_cache


class DatabaseManager:
    """Manager class for Supabase database operations"""
    
    def __init__(self):
        """Initialize Supabase client"""
        try:
            self.client: Client = create_client(
                supabase_url=settings.supabase_url,
                supabase_key=settings.supabase_key
            )
            logger.info("✅ Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
            raise
    
    async def fetch_table_data(
        self,
        table_name: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[str] = None,
        ascending: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch data from a Supabase table
        
        Args:
            table_name: Name of the table to query
            limit: Maximum number of rows to return
            offset: Number of rows to skip
            order_by: Column name to order by
            ascending: Sort order (True for ascending, False for descending)
        
        Returns:
            List of dictionaries containing row data
        """
        try:
            query = self.client.table(table_name).select("*")
            
            if order_by:
                query = query.order(order_by, desc=not ascending)
            
            if limit:
                query = query.limit(limit)
            
            if offset:
                query = query.range(offset, offset + (limit or 1000) - 1)
            
            response = query.execute()
            
            row_count = len(response.data)
            log_data_loading(table_name, row_count, "SUCCESS")
            
            return response.data
        
        except Exception as e:
            logger.error(f"❌ Error fetching data from '{table_name}': {e}")
            log_data_loading(table_name, 0, "FAILED")
            raise
    
    async def fetch_first_row(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch the first row from a table (for health check)
        
        Args:
            table_name: Name of the table
        
        Returns:
            Dictionary containing the first row, or None if table is empty
        """
        try:
            response = self.client.table(table_name).select("*").limit(1).execute()
            
            if response.data and len(response.data) > 0:
                logger.info(f"✅ Health check: '{table_name}' is accessible")
                return response.data[0]
            else:
                logger.warning(f"⚠️ Table '{table_name}' is empty")
                return None
        
        except Exception as e:
            logger.error(f"❌ Health check failed for '{table_name}': {e}")
            return None
    
    async def get_table_count(self, table_name: str) -> int:
        """
        Get total row count from a table
        
        Args:
            table_name: Name of the table
        
        Returns:
            Total number of rows
        """
        try:
            response = self.client.table(table_name).select("*", count="exact").execute()
            count = response.count or 0
            logger.info(f"📊 Table '{table_name}' has {count:,} total rows")
            return count
        
        except Exception as e:
            logger.error(f"❌ Error getting count from '{table_name}': {e}")
            return 0
    
    async def fetch_table_as_dataframe(
        self,
        table_name: str,
        limit: Optional[int] = None,
        batch_size: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch entire table data as pandas DataFrame
        Uses pagination to fetch all rows (Supabase has 1000 row limit per query)
        
        Args:
            table_name: Name of the table
            limit: Optional limit on number of rows (None = fetch all)
            batch_size: Number of rows to fetch per batch (default 1000)
        
        Returns:
            pandas DataFrame containing table data
        """
        try:
            logger.info(f"📥 Fetching all data from '{table_name}' table...")
            
            # Get total count first
            total_count = await self.get_table_count(table_name)
            
            if total_count == 0:
                logger.warning(f"⚠️ No data found in table '{table_name}'")
                return pd.DataFrame()
            
            # Determine how many rows to fetch
            rows_to_fetch = min(limit, total_count) if limit else total_count
            
            # Fetch data in batches
            all_data = []
            offset = 0
            
            while offset < rows_to_fetch:
                current_batch_size = min(batch_size, rows_to_fetch - offset)
                
                logger.debug(f"📥 Fetching batch: rows {offset} to {offset + current_batch_size - 1}")
                
                # Fetch batch
                response = self.client.table(table_name)\
                    .select("*")\
                    .range(offset, offset + current_batch_size - 1)\
                    .execute()
                
                if not response.data:
                    logger.warning(f"⚠️ No data in batch starting at offset {offset}")
                    break
                
                all_data.extend(response.data)
                offset += len(response.data)
                
                # Log progress
                progress = (offset / rows_to_fetch) * 100
                logger.info(f"📊 Progress: {offset:,}/{rows_to_fetch:,} rows ({progress:.1f}%)")
                
                # Safety check: if we got fewer rows than expected, we've reached the end
                if len(response.data) < current_batch_size:
                    break
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            row_count = len(df)
            col_count = len(df.columns)
            
            log_data_loading(table_name, row_count, "SUCCESS")
            logger.info(f"✅ Loaded complete dataset: {row_count:,} rows × {col_count} columns")
            
            # Log column names for debugging
            logger.debug(f"Columns in '{table_name}': {df.columns.tolist()}")
            
            return df
        
        except Exception as e:
            logger.error(f"❌ Error fetching '{table_name}' as DataFrame: {e}")
            log_data_loading(table_name, 0, "FAILED")
            return pd.DataFrame()
    
    async def fetch_unique_marketplaces(self) -> List[str]:
        """
        Fetch unique marketplace names from product_analysis table
        Returns sorted list with 'amazon' first, then alphabetical
        Removes duplicates (case-insensitive) and blank entries
        
        Returns:
            List of unique marketplace names
        """
        try:
            logger.info("📊 Fetching unique marketplaces from product_analysis...")
            
            # Fetch all marketplaces (paginated)
            all_marketplaces = []
            offset = 0
            batch_size = 1000
            
            while True:
                response = self.client.table("product_analysis")\
                    .select("marketplaces")\
                    .range(offset, offset + batch_size - 1)\
                    .execute()
                
                if not response.data:
                    break
                
                all_marketplaces.extend(response.data)
                
                if len(response.data) < batch_size:
                    break
                
                offset += batch_size
            
            if not all_marketplaces:
                logger.warning("⚠️ No marketplaces found in product_analysis table")
                return []
            
            # Extract unique marketplaces (case-insensitive deduplication)
            marketplaces_dict = {}  # Use dict to track original casing
            
            for row in all_marketplaces:
                marketplace = row.get("marketplaces", "").strip()
                
                # Skip empty/blank marketplaces
                if not marketplace:
                    continue
                
                marketplace_lower = marketplace.lower()
                
                # Keep first occurrence (or prefer capitalized version)
                if marketplace_lower not in marketplaces_dict:
                    marketplaces_dict[marketplace_lower] = marketplace
                else:
                    # Prefer capitalized version (e.g., "Flipkart" over "flipkart")
                    existing = marketplaces_dict[marketplace_lower]
                    if marketplace[0].isupper() and not existing[0].isupper():
                        marketplaces_dict[marketplace_lower] = marketplace
            
            # Convert to list
            marketplaces_list = list(marketplaces_dict.values())
            
            # Sort: amazon first, then alphabetical (case-insensitive)
            def sort_key(x):
                if x.lower() == "amazon":
                    return (0, "")
                return (1, x.lower())
            
            marketplaces_list.sort(key=sort_key)
            
            logger.info(f"✅ Found {len(marketplaces_list)} unique marketplaces (after deduplication)")
            logger.debug(f"Marketplaces: {', '.join(marketplaces_list[:10])}{'...' if len(marketplaces_list) > 10 else ''}")
            
            return marketplaces_list
        
        except Exception as e:
            logger.error(f"❌ Error fetching unique marketplaces: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on database tables
        
        Returns:
            Dictionary containing health check results
        """
        logger.info("🏥 Starting database health check...")
        
        tables = ["product_analysis", "normalized_score"]
        results = {
            "status": "healthy",
            "tables": {},
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        for table_name in tables:
            try:
                first_row = await self.fetch_first_row(table_name)
                count = await self.get_table_count(table_name)
                
                results["tables"][table_name] = {
                    "accessible": first_row is not None,
                    "row_count": count,
                    "sample_row": first_row
                }
            
            except Exception as e:
                logger.error(f"❌ Health check failed for table '{table_name}': {e}")
                results["status"] = "unhealthy"
                results["tables"][table_name] = {
                    "accessible": False,
                    "error": str(e)
                }
        
        logger.info(f"🏥 Health check complete: {results['status']}")
        return results


@lru_cache()
def get_supabase_client() -> Client:
    """
    Get Supabase client instance (cached)
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_url = settings.supabase_url
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
    supabase_key = settings.supabase_service_key
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables")
    
    logger.info(f"🔗 Connecting to Supabase: {supabase_url[:30]}...")
    
    client = create_client(supabase_url, supabase_key)
    return client


# Global database manager instance
db_manager = DatabaseManager()
