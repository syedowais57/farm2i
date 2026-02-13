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
import concurrent.futures


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
            
            # Separate SCL band (categorical) - use nearest neighbor to preserve class values
            scl = image.select('SCL').resample('bilinear').reproject(
                crs=projection.crs(),
                scale=10
            ).round().toInt()  # Round to nearest integer to keep categorical values
            
            # Resample all other bands to 10m using bicubic interpolation
            other_bands = image.bandNames().filter(ee.Filter.neq('item', 'SCL'))
            resampled = image.select(other_bands).resample('bicubic').reproject(
                crs=projection.crs(),
                scale=10
            )
            
            # Combine: resampled bands + properly handled SCL
            return resampled.addBands(scl)
        
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
    def calculate_oc(cls, image: ee.Image) -> ee.Image:
        """Calculate OC (Organic Carbon): 3.591 * NDVI"""
        # Ensure NDVI is already calculated and available
        ndvi = image.select('NDVI')
        oc = ndvi.multiply(3.591).rename('OC')
        return image.addBands(oc)
    
    @classmethod
    def calculate_n_index(cls, image: ee.Image) -> ee.Image:
        """Calculate Nitrogen Index: 1.25 * NDRE"""
        ndre = image.select('NDRE')
        n_index = ndre.multiply(1.25).rename('N_Index')
        return image.addBands(n_index)
    
    @classmethod
    def calculate_p_index(cls, image: ee.Image) -> ee.Image:
        """Calculate Phosphorus Index: 0.48 * OC"""
        oc = image.select('OC')
        p_index = oc.multiply(0.48).rename('P_Index')
        return image.addBands(p_index)
        
    @classmethod
    def calculate_k_index(cls, image: ee.Image) -> ee.Image:
        """Calculate Potassium Index: 0.78 * NDVI"""
        ndvi = image.select('NDVI')
        k_index = ndvi.multiply(0.78).rename('K_Index')
        return image.addBands(k_index)
        
    @classmethod
    def calculate_ph_index(cls, image: ee.Image) -> ee.Image:
        """Calculate pH Index: 3.5 + 2.5 * ((B11 - B12) / (B11 + B12))"""
        swir1 = image.select('B11')
        swir2 = image.select('B12')
        ph_ratio = swir1.subtract(swir2).divide(swir1.add(swir2))
        ph_index = ph_ratio.multiply(2.5).add(3.5).rename('pH_Index')
        return image.addBands(ph_index)
    
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
        image = cls.calculate_oc(image)
        image = cls.calculate_n_index(image)
        image = cls.calculate_p_index(image)
        image = cls.calculate_k_index(image)
        image = cls.calculate_ph_index(image)
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
        bands_to_sample = [
            'NDVI', 'GNDVI', 'EVI', 'NDMI', 'NDRE', 'MSAVI', 
            'RECI', 'PSRI', 'MCARI', 'OC', 
            'N_Index', 'P_Index', 'K_Index', 'pH_Index',
            'cloudMask'
        ]
        
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
    
    @staticmethod
    def select_best_per_month(parallel_results: list) -> list:
        """
        From a list of image results, pick the single best image per month.
        Best = lowest cloud cover. On ties, most recent date wins.
        
        Args:
            parallel_results: List of dicts with 'date' (YYYY-MM-DD) and 'cloud_cover' keys.
        Returns:
            Filtered list with at most one entry per year-month, sorted chronologically.
        """
        from collections import defaultdict
        
        monthly_groups = defaultdict(list)
        for res in parallel_results:
            year_month = res['date'][:7]  # 'YYYY-MM'
            monthly_groups[year_month].append(res)
        
        best = []
        for ym in sorted(monthly_groups.keys()):
            images = monthly_groups[ym]
            # Sort: lowest cloud cover first, then most recent date first (descending)
            images.sort(key=lambda x: (x['cloud_cover'], x['date']))
            # Pick lowest cloud; among ties, pick the last date (most recent)
            min_cloud = images[0]['cloud_cover']
            candidates = [img for img in images if img['cloud_cover'] == min_cloud]
            # Most recent among candidates
            candidates.sort(key=lambda x: x['date'], reverse=True)
            best.append(candidates[0])
        
        print(f"   Best-per-month: selected {len(best)} images from {len(parallel_results)} total")
        return best
    
    @classmethod
    def process_polygon(
        cls,
        polygon,
        start_date: str,
        end_date: str,
        indices: List[str] = None,
        padding_meters: int = 10,
        max_cloud_cover: int = 100,
        best_per_month: bool = False
    ) -> Dict:
        """
        Main processing function - equivalent to STAC version
        
        Args:
            polygon: Shapely polygon or GeoDataFrame
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            indices: List of indices to calculate (default: all)
            padding_meters: Buffer around polygon
            best_per_month: If True, return only the best image per month
            
        Returns:
            Dictionary with dates, cloud cover, and index values
        """
        cls.initialize()
        
        # Default to all indices
        if indices is None:
            indices = [
                'NDVI', 'GNDVI', 'EVI', 'NDMI', 'NDRE', 'MSAVI', 
                'RECI', 'PSRI', 'MCARI', 'OC', 
                'N_Index', 'P_Index', 'K_Index', 'pH_Index'
            ]
        
        # Convert polygon to GEE geometry
        ee_geometry = cls.polygon_to_ee_geometry(polygon)
        
        # Buffer for data fetching
        ee_geometry_buffered = ee_geometry.buffer(padding_meters)
        
        # Get image collection
        collection = cls.get_sentinel2_collection(
            ee_geometry_buffered, start_date, end_date, max_cloud_cover
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
        
        def fetch_image_data(i):
            info = image_info[i]
            try:
                image = ee.Image(image_list.get(i))
                data, bounds = cls.sample_rectangle(image, ee_geometry, scale=10)
                if not data:
                    return None
                
                # Pre-calculate median values for each index
                median_results = {}
                for idx in indices:
                    if idx in data:
                        arr = data[idx]
                        if 'cloudMask' in data:
                            mask = data['cloudMask']
                            arr = np.where(mask == 1, arr, np.nan)
                        
                        if np.all(np.isnan(arr)):
                            median_results[idx] = 0
                        else:
                            val = np.nanmedian(arr)
                            median_results[idx] = int(val * 10000) if not np.isnan(val) else 0
                    else:
                        median_results[idx] = 0
                
                return {
                    'date': info['date'],
                    'cloud_cover': info['cloud_cover'],
                    'image_data': {
                        'bands': data,
                        'bounds': bounds
                    },
                    'indices': median_results
                }
            except Exception as e:
                print(f"Error fetching data for {info['date']}: {e}")
                return None

        print(f"Fetching data for {len(image_info)} images in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_index = {executor.submit(fetch_image_data, i): i for i in range(len(image_info))}
            
            parallel_results = []
            for future in concurrent.futures.as_completed(future_to_index):
                res = future.result()
                if res:
                    parallel_results.append(res)
        
        # Sort results by date to maintain chronological order
        parallel_results.sort(key=lambda x: x['date'])
        
        # Apply best-per-month filtering if requested
        if best_per_month:
            parallel_results = cls.select_best_per_month(parallel_results)
        
        for res in parallel_results:
            results['dates'].append(res['date'])
            results['cloud_cover'].append(res['cloud_cover'])
            results['image_data'].append(res['image_data'])
            for idx in indices:
                results['indices'][idx].append(res['indices'].get(idx, 0))

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
