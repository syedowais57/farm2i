#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEE-based Vegetation Indices Calculator

This script uses Google Earth Engine instead of STAC API for satellite data.
"""

import os
import sys
import time
import uuid
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Polygon as MplPolygon
from scipy.ndimage import zoom
import warnings
import requests

from gee_service import GEEService

warnings.filterwarnings("ignore")

# Upload endpoint configuration
UPLOAD_API_ENDPOINT = 'https://farm2i.saibbyweb.com/upload'


def upload_to_server(file_path, field_id, index_type, date_str):
    """Upload a PNG mask file to the remote server."""
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
            
        with open(file_path, 'rb') as f:
            filename = f"{field_id}_{date_str}_{index_type}.png"
            files = {'file': (filename, f, 'image/png')}
            data = {'field_id': field_id, 'index_type': index_type, 'date': date_str}
            
            response = requests.post(UPLOAD_API_ENDPOINT, files=files, data=data, timeout=30)
            
            if response.status_code in [200, 201]:
                result = response.json()
                print(f"Uploaded {filename} successfully")
                return result.get('url', result.get('path', filename))
            else:
                print(f"Upload failed: {response.status_code}")
                return None
                
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return None


def get_color_scale(index_name):
    """Get color scale for each index type"""
    scales = {
        'NDVI': {
            'bounds': [-10000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000],
            'colors': ['#8B0000', '#CD5C5C', '#FFA500', '#FFD700', '#ADFF2F', '#7CFC00', '#32CD32', '#228B22', '#006400', '#004d00']
        },
        'NDMI': {
            'bounds': [-10000, -3000, -1000, 0, 1000, 2000, 3000, 4000, 5000, 7000, 10000],
            'colors': ['#8B4513', '#A0522D', '#CD853F', '#DEB887', '#F5DEB3', '#ADFF2F', '#7CFC00', '#32CD32', '#228B22', '#006400']
        },
        'MSAVI': {
            'bounds': [-10000, 0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000],
            'colors': ['#8B0000', '#CD5C5C', '#FF6347', '#FFA500', '#FFD700', '#ADFF2F', '#7CFC00', '#32CD32', '#228B22', '#006400']
        },
        'NDRE': {
            'bounds': [-10000, 0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000],
            'colors': ['#8B4513', '#CD853F', '#DEB887', '#F0E68C', '#ADFF2F', '#7CFC00', '#32CD32', '#228B22', '#006400', '#004d00']
        },
        'GNDVI': {
            'bounds': [-10000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000],
            'colors': ['#8B0000', '#CD5C5C', '#FF8C00', '#FFD700', '#9ACD32', '#7CFC00', '#32CD32', '#228B22', '#006400', '#004d00']
        },
        'EVI': {
            'bounds': [-10000, -1000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000],
            'colors': ['#8B0000', '#CD5C5C', '#FFA500', '#FFD700', '#ADFF2F', '#7CFC00', '#32CD32', '#228B22', '#006400', '#004d00']
        }
    }
    return scales.get(index_name, scales['NDVI'])


def generate_index_png(index_array, polygon_gdf, output_path, index_name, pixel_scale=100):
    """
    Generate a PNG image for a vegetation index
    
    Args:
        index_array: numpy array of index values (already scaled 0-1)
        polygon_gdf: GeoDataFrame with polygon for clipping
        output_path: Path to save PNG
        index_name: Name of index for color scale
        pixel_scale: Upscaling factor
    """
    # Scale to -10000 to 10000 range
    INDEX = index_array * 10000
    INDEX = np.around(INDEX, 2)
    
    # Get color scale
    scale = get_color_scale(index_name)
    cmap = ListedColormap(scale['colors'])
    norm = BoundaryNorm(scale['bounds'], cmap.N)
    
    # Upsample for smoother output
    INDEX_scaled = zoom(INDEX, pixel_scale, order=1)
    
    print(f"--- {index_name} - Original: {INDEX.shape}, Upscaled: {INDEX_scaled.shape}")
    
    # Create figure
    fig, ax = plt.subplots()
    cmap.set_bad(color='none') # Make NaN transparent
    im = ax.imshow(INDEX_scaled, cmap=cmap, norm=norm, interpolation='bilinear')
    
    # Get polygon for clipping
    img_h, img_w = INDEX_scaled.shape
    
    if polygon_gdf is not None and len(polygon_gdf) > 0:
        poly_coords = polygon_gdf.iloc[0]['geometry']
        
        if hasattr(poly_coords, 'exterior'):
            # Get bounds
            bounds = polygon_gdf.total_bounds
            minx, miny, maxx, maxy = bounds
            
            coords = list(poly_coords.exterior.coords)
            scaled_coords = []
            
            for x, y in coords:
                px = (x - minx) / (maxx - minx) * img_w
                py = img_h - (y - miny) / (maxy - miny) * img_h
                scaled_coords.append([px, py])
            
            # Apply buffer to prevent edge cutting
            PIXEL_BUFFER = 2
            centroid_x = sum(c[0] for c in scaled_coords) / len(scaled_coords)
            centroid_y = sum(c[1] for c in scaled_coords) / len(scaled_coords)
            buffered_coords = []
            
            for px, py in scaled_coords:
                dx = px - centroid_x
                dy = py - centroid_y
                dist = (dx**2 + dy**2) ** 0.5
                if dist > 0:
                    px += (dx / dist) * PIXEL_BUFFER
                    py += (dy / dist) * PIXEL_BUFFER
                buffered_coords.append([px, py])
            
            polygon_patch = MplPolygon(buffered_coords, closed=True, transform=ax.transData)
            im.set_clip_path(polygon_patch)
    
    ax.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.05, transparent=True, dpi=150)
    plt.close(fig)
    
    print(f"💾 Saved: {output_path}")
    return output_path


class FarmVisionModelServiceGEE:
    """GEE-based Farm Vision Model Service"""
    
    @staticmethod
    def calculateINDEXService(start_date, end_date, ApplNo, poly, temp, indices=None, padding_meters=10):
        """
        Calculate vegetation indices using Google Earth Engine
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            ApplNo: Application/field number
            poly: Shapely polygon geometry
            temp: Temporary output path
            indices: List of indices to calculate
            padding_meters: Buffer around polygon
            
        Returns:
            Dictionary with results matching STAC version format
        """
        start_time = time.time()
        
        # Set up paths
        TEMP_PATH = temp
        TIF_PATH = TEMP_PATH + 'tif/'
        PNG_PATH = TEMP_PATH + 'png/'
        JSON_PATH = TEMP_PATH + 'json/'
        signatures_path = TEMP_PATH + 'signatures/'
        
        for path in [TEMP_PATH, TIF_PATH, PNG_PATH, JSON_PATH, signatures_path]:
            os.makedirs(path, exist_ok=True)
        
        # Default indices
        if indices is None:
            indices = ['NDVI', 'GNDVI', 'EVI', 'NDMI', 'NDRE', 'MSAVI']
        
        AppNo = str(ApplNo)
        
        # Create GeoDataFrame
        farm_polygon = gpd.GeoDataFrame([poly], columns=['geometry'], crs='epsg:4326')
        
        print(f"\nGEE PROCESSING:")
        print(f"   Field ID: {AppNo}")
        print(f"   Date Range: {start_date} to {end_date}")
        print(f"   Indices: {indices}")
        print(f"   Padding: {padding_meters}m")

        # Log polygon for debugging (matching STAC version behavior)
        try:
            from matplotlib.patches import Polygon as MplPoly
            fig_poly, ax_poly = plt.subplots(figsize=(8, 8))
            # Extract coordinates for the polygon
            if hasattr(poly, 'exterior'):
                coords = list(poly.exterior.coords)
                polygon_patch = MplPoly(coords, fill=True, facecolor='lightgreen', edgecolor='green', linewidth=2)
                ax_poly.add_patch(polygon_patch)
                ax_poly.autoscale()
                ax_poly.set_aspect('equal')
                ax_poly.set_title(f'Polygon sent to GEE API (Field: {AppNo})')
                debug_poly_path = os.path.join(PNG_PATH, 'DEBUG_polygon_satellite.png')
                plt.savefig(debug_poly_path, dpi=150, bbox_inches='tight')
                plt.close(fig_poly)
                print(f"Saved: DEBUG_polygon_satellite.png to {PNG_PATH}")
        except Exception as e:
            print(f"Warning: Could not save debug polygon: {e}")
        
        # Initialize GEE and process
        try:
            gee_results = GEEService.process_polygon(
                polygon=poly,
                start_date=start_date,
                end_date=end_date,
                indices=indices,
                padding_meters=padding_meters
            )
        except Exception as e:
            print(f"❌ GEE processing failed: {e}")
            return {
                'dates': '[]',
                'cloud': '[]',
                'NDVI': '[]', 'GNDVI': '[]', 'EVI': '[]',
                'NDMI': '[]', 'NDRE': '[]', 'MSAVI': '[]',
                'png_ndvi': [], 'png_gndvi': [], 'png_evi': [],
                'png_ndmi': [], 'png_ndre': [], 'png_msavi': []
            }
        
        dates = gee_results['dates']
        cloud_cover = gee_results['cloud_cover']
        index_values = gee_results['indices']
        image_data_list = gee_results['image_data']
        
        print(f"OK: Found {len(dates)} images")
        
        # Generate PNG files for each date
        png_urls = {idx.lower(): [] for idx in ['NDVI', 'GNDVI', 'EVI', 'NDMI', 'NDRE', 'MSAVI']}
        
        for i, date_str in enumerate(dates):
            current_data = image_data_list[i]
            
            for idx_name in indices:
                idx_lower = idx_name.lower()
                if idx_name in current_data:
                    # Generate filename
                    png_filename = f"{AppNo}_{date_str}_{idx_lower}.png"
                    png_path = os.path.join(PNG_PATH, png_filename)
                    
                    # Get index data
                    arr = current_data[idx_name]
                    
                    # Apply cloud mask if available for PNG generation
                    if 'cloudMask' in current_data:
                        # Mask out clouds (where cloudMask != 1)
                        # We use np.nan to make it transparent in the final PNG
                        arr = np.where(current_data['cloudMask'] == 1, arr, np.nan)
                    
                    # Generate PNG
                    generate_index_png(
                        index_array=arr,
                        polygon_gdf=farm_polygon,
                        output_path=png_path,
                        index_name=idx_name,
                        pixel_scale=5 # Reduced for memory safety
                    )
                    
                    # Upload to server
                    url = upload_to_server(png_path, AppNo, idx_lower, date_str)
                    if url:
                        png_urls[idx_lower].append(url)
                    else:
                        # Fallback to local path if upload fails
                        png_urls[idx_lower].append(png_path)
        
        # Build result dictionary in same format as STAC version
        result = {
            'dates': str(dates),
            'cloud': str(cloud_cover),
        }
        
        for idx in ['NDVI', 'GNDVI', 'EVI', 'NDMI', 'NDRE', 'MSAVI']:
            if idx in index_values:
                result[idx] = str(index_values[idx])
            else:
                result[idx] = '[]'
            
            # Add PNG paths (ensure they map to the lowercase index names for the API)
            result[f'png_{idx.lower()}'] = png_urls.get(idx.lower(), [])
        
        processing_time = round((time.time() - start_time) / 60, 2)
        print(f"\nTIME: Processing completed in {processing_time} minutes")
        
        return result


# Alias for compatibility
FarmVisionModelService = FarmVisionModelServiceGEE


if __name__ == '__main__':
    # Test with sample polygon
    from shapely.geometry import Polygon
    
    coords = [
        (74.81327442321802, 34.02555224449088),
        (74.81373755130355, 34.02566191157703),
        (74.8139038482608, 34.02541293744765),
        (74.81332091484, 34.02524399030139),
        (74.81327442321802, 34.02555224449088)
    ]
    poly = Polygon(coords)
    
    result = FarmVisionModelService.calculateINDEXService(
        start_date='2025-05-01',
        end_date='2025-05-10',
        ApplNo='test_gee',
        poly=poly,
        temp='./image_data/',
        indices=['NDVI', 'EVI']
    )
    
    print("\nResult:")
    print(result)
