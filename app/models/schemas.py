from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="Health status of the service")
    timestamp: str = Field(..., description="Current timestamp")
    environment: str = Field(..., description="Current environment (development/production)")
    tables: Dict[str, Any] = Field(..., description="Status of database tables")


class ProductAnalysisRow(BaseModel):
    """Schema for product_analysis table row"""
    idx: int
    product_category: str
    prompts: str
    response: str
    p_number: str
    p_value: str
    prompt_id: str
    product_name: str
    rank: int
    source: str
    price: str
    price_currency: str
    delivery_fee: str
    delivery_days: str
    extra: str
    marketplaces: str
    card_id: str


class NormalizedScoreRow(BaseModel):
    """Schema for normalized_score table row"""
    idx: int
    product_category: str
    marketplaces: str
    score_sum: int
    score_norm: float
    rank: int


class TableDataResponse(BaseModel):
    """Response model for table data endpoints"""
    table_name: str
    row_count: int
    data: List[Dict[str, Any]]
    timestamp: str


class ErrorResponse(BaseModel):
    """Response model for error responses"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")
