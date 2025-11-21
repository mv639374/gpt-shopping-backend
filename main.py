from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from datetime import datetime

from app.core.config import settings
from app.core.logger import logger, log_api_request
from app.utils.db import db_manager
from app.models.schemas import HealthCheckResponse, ErrorResponse
from app.routers import analytics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info("=" * 80)
    logger.info(f"🚀 Starting {settings.app_title} API")
    logger.info(f"📍 Environment: {settings.environment}")
    logger.info(f"🔧 Debug Mode: {settings.debug}")
    logger.info(f"🌐 CORS Origins: {settings.cors_origins_list}")
    logger.info(f"💾 Redis Enabled: {settings.redis_enabled}")
    logger.info("=" * 80)
    
    # Perform initial health check
    try:
        health_status = await db_manager.health_check()
        logger.info(f"✅ Initial health check: {health_status['status']}")
    except Exception as e:
        logger.error(f"❌ Initial health check failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("=" * 80)
    logger.info(f"🛑 Shutting down {settings.app_title} API")
    logger.info("=" * 80)


# Initialize FastAPI app
app = FastAPI(
    title=settings.app_title,
    description="Backend API for GPT Shopping Analytics Dashboard - AEO/GEO Tracking",
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing"""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration_ms = (time.time() - start_time) * 1000
    log_api_request(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms
    )
    
    return response


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"❌ Unhandled exception: {exc}", exc_info=True)
    
    error_response = ErrorResponse(
        error="Internal Server Error",
        detail=str(exc) if settings.debug else "An unexpected error occurred",
        timestamp=datetime.now().isoformat()
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump()
    )


# Include routers
app.include_router(analytics.router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information
    """
    return {
        "message": f"Welcome to {settings.app_title} API",
        "version": "0.1.0",
        "environment": settings.environment,
        "docs": "/docs",
        "health": "/health",
        "status": "running"
    }


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    Returns database connection status and sample data from tables
    """
    try:
        health_status = await db_manager.health_check()
        
        return HealthCheckResponse(
            status=health_status["status"],
            timestamp=health_status["timestamp"],
            environment=settings.environment,
            tables=health_status["tables"]
        )
    
    except Exception as e:
        logger.error(f"❌ Health check endpoint failed: {e}")
        raise


# Run the application
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        log_level=settings.log_level.lower()
    )
