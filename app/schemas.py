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
        description="Optional list of specific indices to calculate. Valid values: NDVI, NDMI, MSAVI, NDRE, GNDVI, EVI, RECI, PSRI, MCARI, OC, N_Index, P_Index, K_Index, pH_Index. If not provided, all indices are calculated."
    )
    padding_meters: Optional[int] = Field(
        default=10,
        description="Padding in meters around the polygon for fetching satellite data. Higher values fetch more surrounding context. Default: 10 meters."
    )
    max_cloud_cover: Optional[int] = Field(
        default=100,
        description="Maximum cloud cover percentage allowed (0-100). Images exceeding this value will be filtered out. Default: 100."
    )
    best_per_month: Optional[bool] = Field(
        default=False,
        description="If true, return only the single best image per month (lowest cloud cover, most recent on ties). Ideal for monthly reporting."
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
                    "indices": ["NDVI", "EVI"],
                    "best_per_month": True
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
    OC: List[int] = Field(default=[], description="OC values")
    N_Index: List[int] = Field(default=[], description="Nitrogen Index values")
    P_Index: List[int] = Field(default=[], description="Phosphorus Index values")
    K_Index: List[int] = Field(default=[], description="Potassium Index values")
    pH_Index: List[int] = Field(default=[], description="pH Index values")


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
    png_oc: List[str] = []
    png_n_index: List[str] = []
    png_p_index: List[str] = []
    png_k_index: List[str] = []
    png_ph_index: List[str] = []


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
