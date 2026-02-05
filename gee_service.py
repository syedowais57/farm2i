"""
Google Earth Engine Service for Sentinel-2 Vegetation Indices

This module provides GEE-based data fetching and processing for vegetation indices.
Replaces the STAC API approach with server-side GEE computation.
"""

import ee
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import os


# Path to service account key file
SERVICE_ACCOUNT_KEY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'named-magnet-486409-f1-99092699381d.json'
)

# Project ID from the service account
GEE_PROJECT = 'named-magnet-486409-f1'


class GEEService:
    """Google Earth Engine service for Sentinel-2 data"""
    
    # Sentinel-2 Surface Reflectance collection
    COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED'
    
    # Band mapping for indices
    BANDS = {
        'B2': 'blue',      # 10m
        'B3': 'green',     # 10m
        'B4': 'red',       # 10m
        'B5': 'rededge1',  # 20m
        'B8': 'nir',       # 10m
        'B8A': 'nir08',    # 20m
        'B11': 'swir16',   # 20m
        'SCL': 'scl',      # Scene Classification Layer
        'QA60': 'qa60'     # Cloud mask
    }
    
    _initialized = False
    
    @classmethod
    def initialize(cls):
        """Initialize Earth Engine with service account credentials"""
        if cls._initialized:
            return True
        
        try:
            # Load service account credentials from JSON file
            if os.path.exists(SERVICE_ACCOUNT_KEY):
                with open(SERVICE_ACCOUNT_KEY, 'r') as f:
                    key_data = json.load(f)
                
                service_account_email = key_data.get('client_email')
                
                # Create credentials from service account
                credentials = ee.ServiceAccountCredentials(
                    email=service_account_email,
                    key_file=SERVICE_ACCOUNT_KEY
                )
                
                # Initialize with credentials and project
                ee.Initialize(credentials, project=GEE_PROJECT)
                cls._initialized = True
                print(f"OK: GEE initialized with service account: {service_account_email}")
                return True
            else:
                # Fallback to OAuth if no service account key
                print("⚠️  Service account key not found, trying OAuth...")
                ee.Authenticate()
                ee.Initialize(project=GEE_PROJECT)
                cls._initialized = True
                print("OK: GEE initialized with OAuth")
                return True
                
        except Exception as e:
            print(f"ERROR: GEE initialization failed: {e}")
            return False
    
    @classmethod
    def polygon_to_ee_geometry(cls, polygon) -> ee.Geometry:
        """Convert Shapely polygon to Earth Engine geometry"""
        if hasattr(polygon, 'exterior'):
            # Shapely polygon
            coords = list(polygon.exterior.coords)
        elif hasattr(polygon, 'iloc'):
            # GeoDataFrame
            geom = polygon.iloc[0]['geometry']
            coords = list(geom.exterior.coords)
        else:
            coords = polygon
            
        # GEE expects [lon, lat] format
        return ee.Geometry.Polygon([coords])
    
    @classmethod
    def get_sentinel2_collection(
        cls,
        geometry: ee.Geometry,
        start_date: str,
        end_date: str,
        cloud_cover_max: int = 100
    ) -> ee.ImageCollection:
        """
        Get Sentinel-2 image collection for the given parameters
        All bands are resampled to 10m to avoid shape mismatches.
        """
        cls.initialize()
        
        def resample_bands(image):
            # Select 10m bands to get target projection/scale
            projection = image.select('B2').projection()
            
            # Resample all bands to 10m using bicubic interpolation
            return image.resample('bicubic').reproject(
                crs=projection.crs(),
                scale=10
            )

        collection = (
            ee.ImageCollection(cls.COLLECTION)
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max))
            .sort('system:time_start')
            .map(resample_bands)
        )
        
        return collection
    
    @classmethod
    def add_cloud_mask(cls, image: ee.Image) -> ee.Image:
        """Add cloud mask band to image using SCL"""
        scl = image.select('SCL')
        
        # Cloud classes in SCL: 3 (cloud shadow), 8 (cloud medium), 9 (cloud high), 10 (cirrus)
        cloud_mask = (
            scl.neq(3)   # Not cloud shadow
            .And(scl.neq(8))   # Not cloud medium
            .And(scl.neq(9))   # Not cloud high
            .And(scl.neq(10))  # Not cirrus
        )
        
        return image.addBands(cloud_mask.rename('cloudMask'))
    
    @classmethod
    def calculate_ndvi(cls, image: ee.Image) -> ee.Image:
        """Calculate NDVI: (NIR - Red) / (NIR + Red)"""
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)
    
    @classmethod
    def calculate_gndvi(cls, image: ee.Image) -> ee.Image:
        """Calculate GNDVI: (NIR - Green) / (NIR + Green)"""
        gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')
        return image.addBands(gndvi)
    
    @classmethod
    def calculate_evi(cls, image: ee.Image) -> ee.Image:
        """Calculate EVI: 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)"""
        nir = image.select('B8').divide(10000)
        red = image.select('B4').divide(10000)
        blue = image.select('B2').divide(10000)
        
        evi = nir.subtract(red).multiply(2.5).divide(
            nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
        ).rename('EVI')
        
        return image.addBands(evi)
    
    @classmethod
    def calculate_ndmi(cls, image: ee.Image) -> ee.Image:
        """Calculate NDMI: (NIR8A - SWIR) / (NIR8A + SWIR)"""
        ndmi = image.normalizedDifference(['B8A', 'B11']).rename('NDMI')
        return image.addBands(ndmi)
    
    @classmethod
    def calculate_ndre(cls, image: ee.Image) -> ee.Image:
        """Calculate NDRE: (NIR - RedEdge) / (NIR + RedEdge)"""
        ndre = image.normalizedDifference(['B8', 'B5']).rename('NDRE')
        return image.addBands(ndre)
    
    @classmethod
    def calculate_msavi(cls, image: ee.Image) -> ee.Image:
        """Calculate MSAVI: (2*NIR + 1 - sqrt((2*NIR+1)^2 - 8*(NIR-Red))) / 2"""
        nir = image.select('B8').divide(10000)
        red = image.select('B4').divide(10000)
        
        msavi = nir.multiply(2).add(1).subtract(
            nir.multiply(2).add(1).pow(2).subtract(
                nir.subtract(red).multiply(8)
            ).sqrt()
        ).divide(2).rename('MSAVI')
        
        return image.addBands(msavi)

    @classmethod
    def calculate_reci(cls, image: ee.Image) -> ee.Image:
        """Calculate RECI: (NIR / RE1) - 1"""
        reci = image.expression(
            '(NIR / RE1) - 1',
            {
                'NIR': image.select('B8'),
                'RE1': image.select('B5')
            }
        ).rename('RECI')
        return image.addBands(reci)

    @classmethod
    def calculate_psri(cls, image: ee.Image) -> ee.Image:
        """Calculate PSRI: (Red - Blue) / RE2"""
        psri = image.expression(
            '(RED - BLUE) / RE2',
            {
                'RED': image.select('B4'),
                'BLUE': image.select('B2'),
                'RE2': image.select('B6')
            }
        ).rename('PSRI')
        return image.addBands(psri)

    @classmethod
    def calculate_mcari(cls, image: ee.Image) -> ee.Image:
        """Calculate MCARI: ((RE1 - Red) - 0.2*(RE1 - Green)) * (RE1 / Red)"""
        mcari = image.expression(
            '((RE1 - RED) - 0.2 * (RE1 - GREEN)) * (RE1 / RED)',
            {
                'RE1': image.select('B5'),
                'RED': image.select('B4'),
                'GREEN': image.select('B3')
            }
        ).rename('MCARI')
        return image.addBands(mcari)
    
    @classmethod
    def add_all_indices(cls, image: ee.Image) -> ee.Image:
        """Add all vegetation indices to an image"""
        image = cls.add_cloud_mask(image)
        image = cls.calculate_ndvi(image)
        image = cls.calculate_gndvi(image)
        image = cls.calculate_evi(image)
        image = cls.calculate_ndmi(image)
        image = cls.calculate_ndre(image)
        image = cls.calculate_msavi(image)
        image = cls.calculate_reci(image)
        image = cls.calculate_psri(image)
        image = cls.calculate_mcari(image)
        return image
    
    @classmethod
    def get_image_info(cls, collection: ee.ImageCollection) -> List[Dict]:
        """Get metadata for all images in collection"""
        def extract_info(image):
            return ee.Feature(None, {
                'date': ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'),
                'cloud_cover': image.get('CLOUDY_PIXEL_PERCENTAGE'),
                'id': image.id()
            })
        
        features = collection.map(extract_info)
        info = features.getInfo()
        
        return [f['properties'] for f in info['features']]
    
    @classmethod
    def sample_rectangle(
        cls,
        image: ee.Image,
        geometry: ee.Geometry,
        scale: int = 10
    ) -> tuple:
        """
        Sample image data as numpy arrays within a rectangle
        
        Returns:
            Tuple of (data_dict, bounds_dict) where bounds_dict has minx, miny, maxx, maxy
        """
        # Get the bounding box
        bounds = geometry.bounds()
        
        # Get bounds coordinates for return
        bounds_info = bounds.getInfo()
        coords = bounds_info['coordinates'][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        
        # Add 5% buffer to each side to prevent polygon touching frame edges
        lon_range = max(lons) - min(lons)
        lat_range = max(lats) - min(lats)
        buffer_pct = 0.05
        
        bounds_dict = {
            'minx': min(lons) - lon_range * buffer_pct, 
            'miny': min(lats) - lat_range * buffer_pct, 
            'maxx': max(lons) + lon_range * buffer_pct, 
            'maxy': max(lats) + lat_range * buffer_pct
        }
        
        # Create buffered region for sampling
        buffered_bounds = ee.Geometry.Rectangle([
            bounds_dict['minx'], bounds_dict['miny'],
            bounds_dict['maxx'], bounds_dict['maxy']
        ])
        
        # Sample the region
        bands_to_sample = ['NDVI', 'GNDVI', 'EVI', 'NDMI', 'NDRE', 'MSAVI', 'RECI', 'PSRI', 'MCARI', 'cloudMask']
        
        try:
            # Use sampleRectangle for small regions with buffered bounds for padding
            sample = image.select(bands_to_sample).sampleRectangle(
                region=buffered_bounds,
                defaultValue=0
            )
            
            result = {}
            for band in bands_to_sample:
                arr = np.array(sample.get(band).getInfo())
                result[band] = arr
                
            return result, bounds_dict
            
        except Exception as e:
            print(f"sampleRectangle failed, using getDownloadURL: {e}")
            # Fallback to getDownloadURL for larger regions
            return cls._download_as_numpy(image, bounds, bands_to_sample, scale)
    
    @classmethod
    def _download_as_numpy(
        cls,
        image: ee.Image,
        region: ee.Geometry,
        bands: List[str],
        scale: int
    ) -> Dict[str, np.ndarray]:
        """Download image data as numpy arrays using computePixels"""
        try:
            result = {}
            for band in bands:
                # Get the pixels
                pixels = image.select(band).clipToBoundsAndScale(
                    geometry=region,
                    scale=scale
                )
                
                # Convert to numpy
                arr = np.array(pixels.sampleRectangle(region=region).get(band).getInfo())
                result[band] = arr
                
            return result
        except Exception as e:
            print(f"Error downloading as numpy: {e}")
            return {}
    
    @classmethod
    def process_polygon(
        cls,
        polygon,
        start_date: str,
        end_date: str,
        indices: List[str] = None,
        padding_meters: int = 10
    ) -> Dict:
        """
        Main processing function - equivalent to STAC version
        
        Args:
            polygon: Shapely polygon or GeoDataFrame
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            indices: List of indices to calculate (default: all)
            padding_meters: Buffer around polygon
            
        Returns:
            Dictionary with dates, cloud cover, and index values
        """
        cls.initialize()
        
        # Default to all indices
        if indices is None:
            indices = ['NDVI', 'GNDVI', 'EVI', 'NDMI', 'NDRE', 'MSAVI']
        
        # Convert polygon to GEE geometry
        ee_geometry = cls.polygon_to_ee_geometry(polygon)
        
        # Buffer for data fetching
        ee_geometry_buffered = ee_geometry.buffer(padding_meters)
        
        # Get image collection
        collection = cls.get_sentinel2_collection(
            ee_geometry_buffered, start_date, end_date
        )
        
        # Add indices to all images
        collection_with_indices = collection.map(cls.add_all_indices)
        
        # Get image info (dates, cloud cover)
        image_info = cls.get_image_info(collection)
        
        if not image_info:
            print("No images found for the given parameters")
            return {
                'dates': [],
                'cloud_cover': [],
                'indices': {idx: [] for idx in indices}
            }
        
        print(f"Found {len(image_info)} images")
        
        # Process each image
        results = {
            'dates': [],
            'cloud_cover': [],
            'indices': {idx: [] for idx in indices},
            'image_data': []
        }
        
        image_list = collection_with_indices.toList(collection_with_indices.size())
        
        for i, info in enumerate(image_info):
            print(f"Processing image {i+1}/{len(image_info)}: {info['date']}")
            
            try:
                image = ee.Image(image_list.get(i))
                
                # Sample the data - returns (data_dict, bounds_dict)
                data, bounds = cls.sample_rectangle(image, ee_geometry, scale=10)
                
                if not data:
                    print(f"No data for image {info['date']}")
                    continue
                
                results['dates'].append(info['date'])
                results['cloud_cover'].append(info['cloud_cover'])
                
                # Store data with bounds for PNG generation
                results['image_data'].append({
                    'bands': data,
                    'bounds': bounds
                })
                
                # Calculate median values for each index
                for idx in indices:
                    if idx in data:
                        arr = data[idx]
                        # Apply cloud mask if available
                        if 'cloudMask' in data:
                            mask = data['cloudMask']
                            arr = np.where(mask == 1, arr, np.nan)
                        
                        # Calculate median
                        if np.all(np.isnan(arr)):
                            median_val = 0
                        else:
                            val = np.nanmedian(arr)
                            if np.isnan(val):
                                median_val = 0
                            else:
                                median_val = int(val * 10000)
                                
                        results['indices'][idx].append(median_val)
                    else:
                        results['indices'][idx].append(0)
                        
            except Exception as e:
                print(f"Error processing image {info['date']}: {e}")
                # Ensure we don't have mismatched list lengths
                if len(results['dates']) > len(results['indices'][indices[0]]):
                    results['dates'].pop()
                    results['cloud_cover'].pop()
                continue
        
        return results


# Convenience function for testing
def test_gee():
    """Test GEE connection"""
    if GEEService.initialize():
        print("GEE test passed!")
        
        # Quick test query
        test_geom = ee.Geometry.Point([74.8133, 34.0255])
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(test_geom).limit(1)
        count = collection.size().getInfo()
        print(f"Found {count} test images")
        return True
    return False


if __name__ == '__main__':
    test_gee()
