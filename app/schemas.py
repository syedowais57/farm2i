"""
Pydantic schemas for the Sentinel-2 Vegetation Indices API
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date


class Coordinate(BaseModel):
    """A single coordinate point"""
    lat: float = Field(..., description="Latitude (e.g., 34.02555)")
    lng: float = Field(..., description="Longitude (e.g., 74.81327)")


class IndicesRequest(BaseModel):
    """Request model for calculating vegetation indices"""
    coordinates: List[Coordinate] = Field(
        ..., 
        min_length=3,
        description="List of coordinates forming a polygon (minimum 3 points)"
    )
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    field_id: Optional[str] = Field(
        default=None, 
        description="Optional field identifier"
    )
    indices: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific indices to calculate. Valid values: NDVI, NDMI, MSAVI, NDRE, GNDVI, EVI, RECI, PSRI, MCARI. If not provided, all indices are calculated."
    )
    padding_meters: Optional[int] = Field(
        default=10,
        description="Padding in meters around the polygon for fetching satellite data. Higher values fetch more surrounding context. Default: 10 meters."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "coordinates": [
                        {"lat": 34.02555224449088, "lng": 74.81327442321802},
                        {"lat": 34.02566191157703, "lng": 74.81373755130355},
                        {"lat": 34.02541293744765, "lng": 74.8139038482608},
                        {"lat": 34.02524399030139, "lng": 74.81332091484}
                    ],
                    "start_date": "2025-05-01",
                    "end_date": "2025-05-10",
                    "field_id": "testing_payment",
                    "indices": ["NDVI", "EVI"]
                }
            ]
        }
    }


class IndicesData(BaseModel):
    """Individual index values"""
    NDVI: List[int] = Field(default=[], description="NDVI values (scaled by 10000)")
    GNDVI: List[int] = Field(default=[], description="GNDVI values")
    EVI: List[int] = Field(default=[], description="EVI values")
    NDMI: List[int] = Field(default=[], description="NDMI values")
    NDRE: List[int] = Field(default=[], description="NDRE values")
    MSAVI: List[int] = Field(default=[], description="MSAVI values")
    RECI: List[int] = Field(default=[], description="RECI values")
    PSRI: List[int] = Field(default=[], description="PSRI values")
    MCARI: List[int] = Field(default=[], description="MCARI values")


class OutputPaths(BaseModel):
    """Paths to generated output files"""
    png_ndvi: List[str] = []
    png_gndvi: List[str] = []
    png_evi: List[str] = []
    png_ndmi: List[str] = []
    png_ndre: List[str] = []
    png_msavi: List[str] = []
    png_reci: List[str] = []
    png_psri: List[str] = []
    png_mcari: List[str] = []


class IndicesResponse(BaseModel):
    """Response model for vegetation indices calculation"""
    field_id: str
    status: str = Field(default="success")
    processing_time_minutes: float
    dates: List[str]
    cloud_cover: List[float]
    indices: IndicesData
    output_paths: OutputPaths
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model"""
    status: str = "error"
    message: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    service: str = "sentinel-vegetation-indices-api"
    version: str = "1.0.0"
