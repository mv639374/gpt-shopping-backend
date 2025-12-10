from typing import Optional, Dict, Any, List
from supabase import create_client, Client
from app.core.config import settings
from app.core.logger import logger, log_data_loading
import pandas as pd
import os
from functools import lru_cache
import asyncio


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
        Uses a very short timeout to avoid blocking
        
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
    
    async def get_table_count(self, table_name: str, timeout_seconds: int = 30) -> int:
        """
        Get total row count from a table with timeout protection
        For large tables, this might timeout - we'll handle gracefully
        
        Args:
            table_name: Name of the table
            timeout_seconds: Maximum time to wait for count
        
        Returns:
            Total number of rows, or 0 if timeout/error
        """
        try:
            # For large tables, count query can be slow
            # Use limit(0) with count to get just the count, not the data
            response = self.client.table(table_name)\
                .select("*", count="exact")\
                .limit(0)\
                .execute()
            
            count = response.count or 0
            
            if count > 0:
                logger.info(f"📊 Table '{table_name}' has {count:,} total rows")
            else:
                logger.warning(f"⚠️ Table '{table_name}' appears empty or count unavailable")
            
            return count
        
        except Exception as e:
            logger.error(f"❌ Error getting count from '{table_name}': {e}")
            # Don't fail - just return 0 and continue with blind pagination
            return 0
    
    async def fetch_table_as_dataframe(
        self,
        table_name: str,
        limit: Optional[int] = None,
        batch_size: int = 1000,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        Fetch entire table data as pandas DataFrame with production optimizations
        Uses pagination to fetch all rows (Supabase has 1000 row limit per query)
        
        Args:
            table_name: Name of the table
            limit: Optional limit on number of rows (None = fetch all)
            batch_size: Number of rows to fetch per batch (default 1000)
            max_retries: Number of retries for failed batches
        
        Returns:
            pandas DataFrame containing table data
        """
        try:
            logger.info(f"📥 Fetching all data from '{table_name}' table...")
            
            # Try to get total count, but don't fail if it times out
            total_count = await self.get_table_count(table_name, timeout_seconds=30)
            
            if total_count == 0:
                logger.warning(
                    f"⚠️ No count available for '{table_name}'. "
                    f"Will fetch data until exhausted..."
                )
            
            # Determine how many rows to fetch
            if limit and total_count > 0:
                rows_to_fetch = min(limit, total_count)
            elif limit:
                rows_to_fetch = limit
            elif total_count > 0:
                rows_to_fetch = total_count
            else:
                # We don't know the count - fetch until we run out
                rows_to_fetch = None
            
            # Fetch data in batches
            all_data = []
            offset = 0
            consecutive_errors = 0
            
            while True:
                # Calculate current batch size
                if rows_to_fetch:
                    current_batch_size = min(batch_size, rows_to_fetch - offset)
                else:
                    current_batch_size = batch_size
                
                # Safety check: stop if we've exceeded expected rows
                if rows_to_fetch and offset >= rows_to_fetch:
                    logger.info(f"✅ Reached target row count: {rows_to_fetch:,}")
                    break
                
                logger.debug(
                    f"📥 Fetching batch: rows {offset} to {offset + current_batch_size - 1}"
                )
                
                # Fetch batch with retry logic
                batch_data = await self._fetch_batch_with_retry(
                    table_name=table_name,
                    offset=offset,
                    limit=current_batch_size,
                    max_retries=max_retries
                )
                
                # Handle empty batch
                if not batch_data:
                    consecutive_errors += 1
                    
                    if consecutive_errors >= 3:
                        logger.warning(
                            f"⚠️ Got 3 consecutive empty batches at offset {offset}. "
                            f"Assuming end of data."
                        )
                        break
                    
                    # Try next offset
                    offset += current_batch_size
                    continue
                
                # Reset error counter on success
                consecutive_errors = 0
                
                # Add to results
                all_data.extend(batch_data)
                actual_batch_size = len(batch_data)
                offset += actual_batch_size
                
                # Log progress
                if rows_to_fetch:
                    progress = (offset / rows_to_fetch) * 100
                    logger.info(
                        f"📊 Progress: {offset:,}/{rows_to_fetch:,} rows ({progress:.1f}%)"
                    )
                else:
                    logger.info(f"📊 Progress: {offset:,} rows fetched so far")
                
                # Break if we got fewer rows than requested (reached end of table)
                if actual_batch_size < current_batch_size:
                    logger.info(
                        f"✅ Fetched final batch with {actual_batch_size} rows. "
                        f"Total: {offset:,} rows"
                    )
                    break
                
                # Safety limit: prevent infinite loops
                if offset > 1_000_000:  # 1 million row safety limit
                    logger.warning(
                        f"⚠️ Safety limit reached at {offset:,} rows. Stopping fetch."
                    )
                    break
            
            # Check if we got any data
            if not all_data:
                logger.warning(f"⚠️ No data found in table '{table_name}'")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            row_count = len(df)
            col_count = len(df.columns)
            
            log_data_loading(table_name, row_count, "SUCCESS")
            logger.info(f"✅ Loaded complete dataset: {row_count:,} rows × {col_count} columns")
            
            # Log column names for debugging
            logger.info(f"📋 Columns in '{table_name}': {df.columns.tolist()}")
            
            return df
        
        except Exception as e:
            logger.error(f"❌ Error fetching '{table_name}' as DataFrame: {e}")
            log_data_loading(table_name, 0, "FAILED")
            return pd.DataFrame()
    
    async def _fetch_batch_with_retry(
        self,
        table_name: str,
        offset: int,
        limit: int,
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Fetch a single batch of data with retry logic
        
        Args:
            table_name: Name of the table
            offset: Starting row offset
            limit: Number of rows to fetch
            max_retries: Maximum number of retry attempts
        
        Returns:
            List of row dictionaries
        """
        for attempt in range(max_retries):
            try:
                response = self.client.table(table_name)\
                    .select("*")\
                    .range(offset, offset + limit - 1)\
                    .execute()
                
                return response.data if response.data else []
            
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        f"⚠️ Batch fetch failed (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {wait_time}s... Error: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"❌ Batch fetch failed after {max_retries} attempts at offset {offset}: {e}"
                    )
                    return []
    
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
            max_batches = 1000  # Safety limit
            batch_count = 0
            
            while batch_count < max_batches:
                try:
                    response = self.client.table("product_analysis")\
                        .select("marketplaces")\
                        .range(offset, offset + batch_size - 1)\
                        .execute()
                    
                    if not response.data or len(response.data) == 0:
                        break
                    
                    all_marketplaces.extend(response.data)
                    
                    if len(response.data) < batch_size:
                        break
                    
                    offset += batch_size
                    batch_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ Error fetching marketplace batch at offset {offset}: {e}")
                    break
            
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
                # Test basic access with first row
                first_row = await self.fetch_first_row(table_name)
                
                # Try to get count, but don't fail if it times out
                count = await self.get_table_count(table_name, timeout_seconds=30)
                
                results["tables"][table_name] = {
                    "accessible": first_row is not None,
                    "row_count": count if count > 0 else "unknown",
                    "columns": list(first_row.keys()) if first_row else [],
                    "status": "ok" if first_row else "empty"
                }
            
            except Exception as e:
                logger.error(f"❌ Health check failed for table '{table_name}': {e}")
                results["status"] = "degraded"
                results["tables"][table_name] = {
                    "accessible": False,
                    "error": str(e),
                    "status": "error"
                }
        
        logger.info(f"🏥 Health check complete: {results['status']}")
        return results


@lru_cache()
def get_supabase_client() -> Client:
    """
    Get Supabase client instance (cached)
    """
    supabase_url = settings.supabase_url
    supabase_key = settings.supabase_service_key
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables")
    
    logger.info(f"🔗 Connecting to Supabase: {supabase_url[:30]}...")
    
    client = create_client(supabase_url, supabase_key)
    return client


# Global database manager instance
db_manager = DatabaseManager()