from typing import Optional, Any
import json
from app.core.config import settings
from app.core.logger import logger

# Redis client - initialize only if enabled
redis_client = None

if settings.redis_enabled:
    try:
        import redis
        redis_client = redis.Redis(
            host=settings.upstash_redis_url.split("//")[1].split(":")[0],
            port=int(settings.upstash_redis_url.split(":")[-1].split("/")[0]),
            password=settings.upstash_redis_token,
            decode_responses=True,
            socket_connect_timeout=5,
        )
        logger.info("✅ Redis cache initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Redis: {e}")
        redis_client = None


async def get_cache(key: str) -> Optional[Any]:
    """
    Get value from Redis cache
    
    Args:
        key: Cache key
    
    Returns:
        Cached value or None
    """
    if not redis_client:
        return None
    
    try:
        value = redis_client.get(key)
        if value:
            logger.debug(f"✅ Cache HIT: {key}")
            return json.loads(value)
        logger.debug(f"❌ Cache MISS: {key}")
        return None
    except Exception as e:
        logger.error(f"❌ Cache GET error for {key}: {e}")
        return None


async def set_cache(key: str, value: Any, ttl: int = None) -> bool:
    """
    Set value in Redis cache
    
    Args:
        key: Cache key
        value: Value to cache
        ttl: Time to live in seconds (default from settings)
    
    Returns:
        True if successful, False otherwise
    """
    if not redis_client:
        return False
    
    try:
        ttl = ttl or settings.cache_ttl
        redis_client.setex(
            key,
            ttl,
            json.dumps(value)
        )
        logger.debug(f"✅ Cache SET: {key} (TTL: {ttl}s)")
        return True
    except Exception as e:
        logger.error(f"❌ Cache SET error for {key}: {e}")
        return False


async def delete_cache(key: str) -> bool:
    """
    Delete key from Redis cache
    
    Args:
        key: Cache key to delete
    
    Returns:
        True if successful, False otherwise
    """
    if not redis_client:
        return False
    
    try:
        redis_client.delete(key)
        logger.debug(f"✅ Cache DELETE: {key}")
        return True
    except Exception as e:
        logger.error(f"❌ Cache DELETE error for {key}: {e}")
        return False
