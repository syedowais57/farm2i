"""
FastAPI server for Sentinel-2 Vegetation Indices Calculation (GEE Version)

Endpoints:
- POST /api/v1/calculate-indices - Calculate vegetation indices for a polygon
- GET /health - Health check

Data Source: Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED)
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import uuid
from typing import Optional
from shapely.geometry import Polygon

from app.schemas import (
    IndicesRequest, 
    IndicesResponse, 
    IndicesData,
    OutputPaths,
    ErrorResponse, 
    HealthResponse
)
# Import GEE version instead of STAC version
from script_gee import FarmVisionModelService

# Create FastAPI app
app = FastAPI(
    title="Sentinel-2 Vegetation Indices API (GEE)",
    description="Calculate vegetation indices (NDVI, GNDVI, EVI, NDMI, NDRE, MSAVI) from Sentinel-2 via Google Earth Engine",

    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def coordinates_to_polygon(coordinates: list) -> Polygon:
    """Convert list of Coordinate objects to Shapely Polygon"""
    # Extract (lng, lat) tuples - Shapely uses (x, y) = (lng, lat)
    points = [(coord.lng, coord.lat) for coord in coordinates]
    # Close the polygon if not already closed
    if points[0] != points[-1]:
        points.append(points[0])
    return Polygon(points)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse()


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - redirects to docs"""
    return {
        "message": "Sentinel-2 Vegetation Indices API",
        "docs": "/docs",
        "health": "/health"
    }


@app.post(
    "/api/v1/calculate-indices",
    response_model=IndicesResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    tags=["Indices"]
)
async def calculate_indices(request: IndicesRequest):
    """
    Calculate vegetation indices for a given polygon and date range.
    
    This endpoint calculates 6 vegetation indices:
    - **NDVI**: Normalized Difference Vegetation Index
    - **GNDVI**: Green Normalized Difference Vegetation Index  
    - **EVI**: Enhanced Vegetation Index
    - **NDMI**: Normalized Difference Moisture Index
    - **NDRE**: Normalized Difference Red Edge Index
    - **MSAVI**: Modified Soil Adjusted Vegetation Index
    
    **Note**: Processing takes approximately 10-15 minutes depending on the date range.
    """
    start_time = time.time()
    
    # Generate field_id if not provided
    field_id = request.field_id or f"field_{uuid.uuid4().hex[:8]}"
    
    try:
        # Convert coordinates to Shapely Polygon
        polygon = coordinates_to_polygon(request.coordinates)
        
        # Validate polygon
        if not polygon.is_valid:
            raise HTTPException(
                status_code=400,
                detail="Invalid polygon geometry. Please check your coordinates."
            )
        
        # Set output directory
        temp_path = './image_data/'
        
        # Validate and prepare indices list
        valid_indices = {'NDVI', 'NDMI', 'MSAVI', 'NDRE', 'GNDVI', 'EVI', 'RECI', 'PSRI', 'MCARI'}
        requested_indices = request.indices
        
        if requested_indices:
            # Validate requested indices
            invalid = set(requested_indices) - valid_indices
            if invalid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid indices: {invalid}. Valid options: {valid_indices}"
                )
        else:
            # Default: calculate all indices
            requested_indices = list(valid_indices)
        
        # Call the service
        result = FarmVisionModelService.calculateINDEXService(
            start_date=request.start_date,
            end_date=request.end_date,
            ApplNo=field_id,
            poly=polygon,
            temp=temp_path,
            indices=requested_indices,
            padding_meters=request.padding_meters or 10
        )
        
        # Calculate processing time
        processing_time = round((time.time() - start_time) / 60, 2)
        
        # Parse the result
        # The result dict has string representations of lists, need to parse them
        import ast
        
        def parse_list(s):
            """Safely parse string representation of list"""
            try:
                return ast.literal_eval(s) if isinstance(s, str) else s
            except:
                return []
        
        dates = parse_list(result.get('dates', []))
        cloud_cover = parse_list(result.get('cloud', []))
        
        # Build response
        response = IndicesResponse(
            field_id=field_id,
            status="success",
            processing_time_minutes=processing_time,
            dates=dates,
            cloud_cover=cloud_cover,
            indices=IndicesData(
                NDVI=parse_list(result.get('NDVI', [])),
                GNDVI=parse_list(result.get('GNDVI', [])),
                EVI=parse_list(result.get('EVI', [])),
                NDMI=parse_list(result.get('NDMI', [])),
                NDRE=parse_list(result.get('NDRE', [])),
                MSAVI=parse_list(result.get('MSAVI', [])),
                RECI=parse_list(result.get('RECI', [])),
                PSRI=parse_list(result.get('PSRI', [])),
                MCARI=parse_list(result.get('MCARI', []))
            ),
            output_paths=OutputPaths(
                png_ndvi=result.get('png_ndvi', []),
                png_gndvi=result.get('png_gndvi', []),
                png_evi=result.get('png_evi', []),
                png_ndmi=result.get('png_ndmi', []),
                png_ndre=result.get('png_ndre', []),
                png_msavi=result.get('png_msavi', []),
                png_reci=result.get('png_reci', []),
                png_psri=result.get('png_psri', []),
                png_mcari=result.get('png_mcari', [])
            ),
            message=f"Successfully calculated indices for {len(dates)} dates"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


# Run with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
