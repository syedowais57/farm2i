#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:21:50 2024

@author: santhosh
"""

# module import
import os.path
import os,sys
import json
import time
import pandas as pd
import numpy as np
import rasterio
from rasterstats.io import Raster
import geopandas as gpd
from multiprocessing import Pool
import matplotlib.pyplot as plt
from pystac_client import Client
from rasterio.mask import mask
import uuid
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from azure.storage.blob import BlobClient
# Define Global Variable
global all_data_df, farm_polygon, farm_polygon1, AppNo,TEMP_PATH,TIF_PATH,PNG_PATH,JSON_PATH,signatures_path
import matplotlib.pyplot as plt
import warnings
import base64
from matplotlib.colors import ListedColormap, BoundaryNorm
import binascii
import requests
from shapely import wkb
from scipy.ndimage import zoom
warnings.filterwarnings("ignore")

SHP = '.shp'
TIF = '.tif'
sentinal2_element84_url = "https://earth-search.aws.element84.com/v1"
sentinal2_collection = "sentinel-2-l2a"




class FarmVisionModelService:
    @staticmethod
    def series_from(item):
            item_dict = {'item_id': item.id,
                         'Date': item.properties['datetime'][0:10],
                         'R': item.assets['red'].href,
                         'N': item.assets['nir'].href,
                         'G': item.assets['green'].href,
                         'B': item.assets['blue'].href,
                         'N8A': item.assets['nir08'].href,
                         'RE1': item.assets['rededge1'].href,
                         'SWIR': item.assets['swir16'].href,
                         'SCL' : item.assets['scl'].href,
                         }
            return pd.Series(item_dict)
             
         
    def Clip_raster(raster,df, Clipped_raster):
            with rasterio.open(raster) as src:
                df = df.to_crs(src.crs)
                shapes = [json.loads(df.to_json())['features'][0]['geometry']]
                
                # SAVE: Polygon shape that goes to satellite
                import matplotlib.pyplot as plt
                from matplotlib.patches import Polygon as MplPoly
                fig_poly, ax_poly = plt.subplots(figsize=(8, 8))
                coords = shapes[0]['coordinates'][0]
                polygon = MplPoly(coords, fill=True, facecolor='lightgreen', edgecolor='green', linewidth=2)
                ax_poly.add_patch(polygon)
                ax_poly.autoscale()
                ax_poly.set_aspect('equal')
                ax_poly.set_title('Polygon sent to Satellite API')
                plt.savefig('./image_data/png/DEBUG_polygon_satellite.png', dpi=150, bbox_inches='tight')
                plt.close(fig_poly)
                print("Saved: DEBUG_polygon_satellite.png")
                
                out_image, out_transform = mask(src, shapes, all_touched=True, crop=True)
                out_meta = src.meta
            out_meta.update({"driver": "GTIFF",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform,
                             "compress": 'DEFLATE'})
            with rasterio.open(Clipped_raster, "w", **out_meta) as dest:
                dest.write(out_image)
          
    def clip_ras2ras(source, reference,outtif):
            with rasterio.open(source) as src:
                # Open the reference raster
                with rasterio.open(reference) as ref:
                    # Define the transformation and dimensions from the reference raster
                    new_transform = from_origin(ref.bounds.left, ref.bounds.top, src.res[0], src.res[1])
                    new_height = ref.shape[0]
                    new_width = ref.shape[1]
                    # Create the output dataset
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'transform': new_transform,
                        'height': new_height,
                        'width': new_width,
                        'dtype': 'float32',  # Adjust data type as needed
                        'count': 1  # Assuming a single-band raster
                    })
                    with rasterio.open(outtif, 'w', **kwargs) as dst:
                        # Resample the data using the reference raster's extent and transformation
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=rasterio.band(dst, 1),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=new_transform,
                            dst_crs=ref.crs,
                            resampling=Resampling.bilinear  # Set the resampling method
                        )
        
    @staticmethod
    def indices_to_tif(raster_file, indices_value, destination_path):
        """
            This function is used to generate png image of indices values(array)
        :param raster_file: source file that indicates that PNG file will generate into same format
        :type raster_file: tif file
        :param indices_value: array value
        :type indices_value: np.array()
        :param destination_path: directory destination path
        :type destination_path: str
        :return: 0/1
        :rtype: int
        """
        with rasterio.open(raster_file) as dataset:
                meta_data = dataset.meta
                #logger.info(f"file: {raster_file}, crs: {dataset.crs}")
        new_dataset = rasterio.open(
                destination_path,
                'w',
                driver='GTiff',
                compress="DEFLATE",
                height=indices_value.shape[1],
                width=indices_value.shape[2],
                count=indices_value.shape[0],
                dtype=indices_value.dtype,
                crs=meta_data['crs'],
                nodata=0,
                transform=meta_data['transform']
            )
        new_dataset.write(indices_value)
        new_dataset.close()
        return 1
    
    @staticmethod
    def cliprs2aoi(band,polygon):
         with Raster(band) as raster_obj:
                    shape = polygon.to_crs(raster_obj.src.crs)
                    geom_bounds = shape.iloc[0]['geometry'].bounds
                    geom = shape.iloc[0]['geometry']
                    raster_subset = raster_obj.read(bounds=geom_bounds)
                    polygon_mask = rasterio.features.geometry_mask(geometries=[geom],
                                                                   out_shape=(raster_subset.shape[0],
                                                                              raster_subset.shape[1]),
                                                                   transform=raster_subset.affine,
                                                                   all_touched=False,
                                                                   invert=True)
                    return raster_subset.array * polygon_mask, raster_subset, raster_obj

    @staticmethod
    def png2base64(png):
        with open(png, 'rb') as f:
            img = f.read()
        pngencode = str(base64.b64encode(img))

        return pngencode
         
    
    @staticmethod
    def s2indexcompute(st_date,filtered_data,s2_index):
            #t2 = int(time.time())
            Band1 = []
            Band2 = []
            Band3 = []  # For EVI (Blue band)
            cloud = []
            '''
            10m : R(B04), G(B03), B(B02), NIR(B08)
            20m : Red Edge(B05), SWIR(B11), B8A
            # Indices formulae
            NDVI = NIR(B08) - RED(B04) / NIR(B08) + RED(B04)
            #NDWI = GREEN(B03) - NIR(B08)/GREEN(B03) + NIR(B08)
            NDMI = NIR(B8A) - SWIR(B11) / NIR(B8A) + SWIR(B11)
            NDRE = NIR(B08) - RED EDGE(B05) / NIR(B08) + RED EDGE(B05)
            CHL   = RED EDGE(B8a) - Red(B04) / RED EDGE(B8a) + Red(B04)
            #EVI = 2.5 * ((B8 – B4) / (B8 + 6 * B4 – 7.5 * B2 + 1))
            '''
            for index, row in filtered_data.iterrows():
                if (s2_index == 'NDVI') | (s2_index=='MSAVI'):
                    Band1_href = row['R']
                    Band2_href = row['N']
                    cloud_href = row['SCL']
                    cloudclip,raster_subset,raster_obj = FarmVisionModelService.cliprs2aoi(cloud_href,farm_polygon)
                    Band1clip,raster_subset,raster_obj = FarmVisionModelService.cliprs2aoi(Band1_href,farm_polygon)
                    Band2clip,raster_subset,raster_obj = FarmVisionModelService.cliprs2aoi(Band2_href,farm_polygon)
                    cloudclip = np.repeat(cloudclip,2, axis=1).repeat(2, axis=0)
                    shape = Band2clip.shape
                    cloudclip = cloudclip[:shape[0],:shape[1]]
                    Band1.append(Band1clip)
                    Band2.append(Band2clip)
                    cloud.append(cloudclip)
                elif s2_index == 'NDMI':
                    Band1_href = row['SWIR']
                    Band2_href = row['N8A']
                    Band3_href = row['N']
                    cloud_href = row['SCL']
                    cloudclip,raster_subset,raster_obj = FarmVisionModelService.cliprs2aoi(cloud_href,farm_polygon)
                    Band1clip,raster_subset,raster_obj = FarmVisionModelService.cliprs2aoi(Band1_href,farm_polygon)
                    Band1clip = np.repeat(Band1clip,2, axis=1).repeat(2, axis=0)
                    Band2clip,raster_subset,raster_obj = FarmVisionModelService.cliprs2aoi(Band2_href,farm_polygon)
                    Band2clip = np.repeat(Band2clip,2, axis=1).repeat(2, axis=0)
                    Band3clip,raster_subset,raster_obj = FarmVisionModelService.cliprs2aoi(Band3_href,farm_polygon)
                    Band3clip = np.repeat(Band3clip,2, axis=1).repeat(2, axis=0)
                    shape = Band3clip.shape
                    Band1clip = Band1clip[:shape[0],:shape[1]]
                    Band2clip = Band2clip[:shape[0],:shape[1]]
                    cloudclip = np.repeat(cloudclip,2, axis=1).repeat(2, axis=0)
                    cloudclip = cloudclip[:shape[0],:shape[1]]
                    Band1.append(Band1clip)
                    Band2.append(Band2clip)
                    cloud.append(cloudclip)
                elif s2_index == 'NDRE':
                    # NDRE = (NIR - Red Edge) / (NIR + Red Edge)
                    # NIR (B08) is 10m, RE1 (B05) is 20m - needs resampling
                    Band1_href = row['RE1']  # Red Edge (B05) - 20m
                    Band2_href = row['N']    # NIR (B08) - 10m
                    cloud_href = row['SCL']
                    cloudclip,raster_subset,raster_obj = FarmVisionModelService.cliprs2aoi(cloud_href,farm_polygon)
                    Band1clip,raster_subset,raster_obj = FarmVisionModelService.cliprs2aoi(Band1_href,farm_polygon)
                    # Upsample RE1 from 20m to 10m
                    Band1clip = np.repeat(Band1clip,2, axis=1).repeat(2, axis=0)
                    Band2clip,raster_subset,raster_obj = FarmVisionModelService.cliprs2aoi(Band2_href,farm_polygon)
                    shape = Band2clip.shape
                    Band1clip = Band1clip[:shape[0],:shape[1]]
                    # SCL is 20m, so upsample to match 10m bands
                    cloudclip = np.repeat(cloudclip,2, axis=1).repeat(2, axis=0)
                    cloudclip = cloudclip[:shape[0],:shape[1]]
                    Band1.append(Band1clip)
                    Band2.append(Band2clip)
                    cloud.append(cloudclip)
                elif s2_index == 'GNDVI':
                    # GNDVI = (NIR - Green) / (NIR + Green)
                    # Both NIR (B08) and Green (B03) are at 10m resolution
                    Band1_href = row['G']   # Green (B03) - 10m
                    Band2_href = row['N']   # NIR (B08) - 10m
                    cloud_href = row['SCL']
                    cloudclip,raster_subset,raster_obj = FarmVisionModelService.cliprs2aoi(cloud_href,farm_polygon)
                    Band1clip,raster_subset,raster_obj = FarmVisionModelService.cliprs2aoi(Band1_href,farm_polygon)
                    Band2clip,raster_subset,raster_obj = FarmVisionModelService.cliprs2aoi(Band2_href,farm_polygon)
                    # SCL is 20m, so upsample to match 10m bands
                    cloudclip = np.repeat(cloudclip,2, axis=1).repeat(2, axis=0)
                    shape = Band2clip.shape
                    cloudclip = cloudclip[:shape[0],:shape[1]]
                    Band1.append(Band1clip)
                    Band2.append(Band2clip)
                    cloud.append(cloudclip)
                elif s2_index == 'EVI':
                    # EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
                    # All bands (B08, B04, B02) are at 10m resolution
                    Band1_href = row['R']   # Red (B04) - 10m
                    Band2_href = row['N']   # NIR (B08) - 10m
                    Band3_href = row['B']   # Blue (B02) - 10m
                    cloud_href = row['SCL']
                    cloudclip,raster_subset,raster_obj = FarmVisionModelService.cliprs2aoi(cloud_href,farm_polygon)
                    Band1clip,raster_subset,raster_obj = FarmVisionModelService.cliprs2aoi(Band1_href,farm_polygon)
                    Band2clip,raster_subset,raster_obj = FarmVisionModelService.cliprs2aoi(Band2_href,farm_polygon)
                    Band3clip,raster_subset,raster_obj = FarmVisionModelService.cliprs2aoi(Band3_href,farm_polygon)
                    # SCL is 20m, so upsample to match 10m bands
                    cloudclip = np.repeat(cloudclip,2, axis=1).repeat(2, axis=0)
                    shape = Band2clip.shape
                    cloudclip = cloudclip[:shape[0],:shape[1]]
                    Band1.append(Band1clip)
                    Band2.append(Band2clip)
                    Band3.append(Band3clip)
                    cloud.append(cloudclip)
            minxlist = [i.shape[0] for i in Band1]
            minylist = [i.shape[1] for i in Band1]
            min_x = np.min(minxlist)
            min_y= np.min(minylist)
            cloud = [i[:min_x,:min_y] for i in cloud]
            Band1 = [i[:min_x,:min_y] for i in Band1]
            Band2 = [i[:min_x,:min_y] for i in Band2]
            Band2 = np.dstack(Band2)
            Band2 = np.rollaxis(Band2, -1)
            Band2 = Band2.astype(np.int16)
            Band2 = Band2+1000
            Band1 = np.dstack(Band1)
            Band1 = np.rollaxis(Band1, -1)
            Band1 = Band1.astype(np.int16)
            Band1 = Band1+1000
            cloud = np.dstack(cloud)
            cloud = np.rollaxis(cloud, -1)
            cloud = cloud.min(axis=0)
            if (s2_index == 'NDVI') | (s2_index == 'NDMI') | (s2_index == 'NDWI') | (s2_index == 'NDRE') | (s2_index == 'GNDVI'):
                calculated_index = (Band2 - Band1) / (Band2 + Band1)
            elif s2_index == 'MSAVI':
                a = (2*Band2*0.0001+1)**2
                b = 8*(Band2*0.0001-Band1*0.0001)
                calculated_index = (2*Band2*0.0001 + 1 -np.sqrt(a-b))/2
            elif s2_index == 'EVI':
                # EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
                # Process Band3 for EVI
                Band3 = [i[:min_x,:min_y] for i in Band3]
                Band3 = np.dstack(Band3)
                Band3 = np.rollaxis(Band3, -1)
                Band3 = Band3.astype(np.int16)
                Band3 = Band3+1000
                # Scale to reflectance (divide by 10000)
                nir = Band2 * 0.0001
                red = Band1 * 0.0001
                blue = Band3 * 0.0001
                calculated_index = 2.5 * ((nir - red) / (nir + 6*red - 7.5*blue + 1))
            max_index = calculated_index.max(axis=0)
            max_index[max_index > 1] = 1
            destination = TEMP_PATH + s2_index+'_'+str(uuid.uuid4())+TIF
            cloudpath = TEMP_PATH + 'cloud'+'_'+str(uuid.uuid4())+TIF
            shape = max_index.shape
            new_dataset = rasterio.open(destination, 'w', driver='GTiff', compress="DEFLATE",
                                        height=shape[0], width=shape[1], count=1, crs=raster_obj.src.crs,
                                        dtype=max_index.dtype, transform=raster_subset.affine, )
            max_index = max_index.reshape(1,shape[0],shape[1])
            new_dataset.write(max_index)
            new_dataset.close()
            new_dataset = rasterio.open(cloudpath, 'w', driver='GTiff', compress="DEFLATE",
                                        height=shape[0], width=shape[1], count=1, crs=raster_obj.src.crs,
                                        dtype=max_index.dtype, transform=raster_subset.affine, )
            cloud = cloud.reshape(1,shape[0],shape[1])
            new_dataset.write(cloud)
            new_dataset.close()
            output_surface = TEMP_PATH + 'surface_'+str(uuid.uuid4())+TIF
            Clipped_raster = TEMP_PATH + 'clipped_'+str(uuid.uuid4())+TIF
            cloud_clip = TEMP_PATH + 'cloudclip_'+str(uuid.uuid4())+TIF
            temp=rasterio.open(destination)
            temp_index = temp.read()+100
            shape2 = temp_index.shape
            new_dataset = rasterio.open(output_surface, 'w', driver='GTiff', compress="DEFLATE",
                                        height=shape2[1], width=shape2[2], count=1, crs=temp.meta['crs'],
                                        dtype=max_index.dtype, transform=temp.meta['transform'], )
            new_dataset.write(temp_index)
            new_dataset.close()
            FarmVisionModelService.Clip_raster(output_surface, farm_polygon1, Clipped_raster)
            FarmVisionModelService.Clip_raster(cloudpath, farm_polygon1, cloud_clip)
            cloud = rasterio.open(cloud_clip).read(1)
            cloud_value = round((((cloudclip==9) | (cloudclip==3)).sum()/(cloudclip!=0).sum())*100,2)
            max_index = rasterio.open(Clipped_raster).read(1)
            max_index[(cloud==9) | (cloud==3)]=0
            max_index = max_index-100
            max_index[max_index==-100] = np.nan
            nan_median = np.nan_to_num(np.nanmedian(max_index))
            median_index = int(nan_median*10000)
            INDEX = max_index*10000
            INDEX = np.around(INDEX,2)
            png_path = PNG_PATH+ AppNo+ '_'+str(st_date)[:10] + '_{}.png'.format(s2_index)
            
            # Index-specific color scales
            if s2_index == 'NDVI':
                # Vegetation: Red (bare) -> Yellow (sparse) -> Green (healthy)
                bounds = [-10000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000]
                colors = ['#8B0000', '#CD5C5C', '#FFA500', '#FFD700', '#ADFF2F', '#7CFC00', '#32CD32', '#228B22', '#006400', '#004d00']
            elif s2_index == 'NDMI':
                # Moisture: Brown (dry) -> Yellow -> Green (moist)
                bounds = [-10000, -3000, -1000, 0, 1000, 2000, 3000, 4000, 5000, 7000, 10000]
                colors = ['#8B4513', '#A0522D', '#CD853F', '#DEB887', '#F5DEB3', '#ADFF2F', '#7CFC00', '#32CD32', '#228B22', '#006400']
            elif s2_index == 'MSAVI':
                # Soil-adjusted vegetation: similar to NDVI
                bounds = [-10000, 0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000]
                colors = ['#8B0000', '#CD5C5C', '#FF6347', '#FFA500', '#FFD700', '#ADFF2F', '#7CFC00', '#32CD32', '#228B22', '#006400']
            elif s2_index == 'NDRE':
                # Red Edge vegetation: Brown (stressed) -> Yellow -> Green (healthy)
                bounds = [-10000, 0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000]
                colors = ['#8B4513', '#CD853F', '#DEB887', '#F0E68C', '#ADFF2F', '#7CFC00', '#32CD32', '#228B22', '#006400', '#004d00']
            elif s2_index == 'GNDVI':
                # Green NDVI: Red (bare) -> Yellow (sparse) -> Green (healthy)
                bounds = [-10000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000]
                colors = ['#8B0000', '#CD5C5C', '#FF8C00', '#FFD700', '#9ACD32', '#7CFC00', '#32CD32', '#228B22', '#006400', '#004d00']
            elif s2_index == 'EVI':
                # Enhanced Vegetation Index: similar scale to NDVI but different range
                bounds = [-10000, -1000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000]
                colors = ['#8B0000', '#CD5C5C', '#FFA500', '#FFD700', '#ADFF2F', '#7CFC00', '#32CD32', '#228B22', '#006400', '#004d00']
            else:
                # Default fallback
                bounds = [-10000, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 10000]
                colors = ['#ad0028', '#c5142a', '#e02d2c', '#ef4c3a', '#fe6c4a', '#ff8d5a', '#ffab69', '#ffc67d', '#ffe093', '#ffefab']
            
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(bounds, cmap.N)
            
            # Upsample to get more pixels
            PIXEL_SCALE = 100  # Higher = finer pixels
            INDEX_scaled = zoom(INDEX, PIXEL_SCALE, order=1)  # order=1 = bilinear
            
            # Create figure with exact polygon shape masking
            from matplotlib.patches import Polygon as MplPolygon
            from matplotlib.path import Path
            
            fig, ax = plt.subplots()
            im = ax.imshow(INDEX_scaled, cmap=cmap, norm=norm, interpolation='nearest')
            
            # Get polygon coordinates and scale to image coordinates
            # Reproject farm_polygon1 to the raster's CRS for correct coordinate matching
            with rasterio.open(Clipped_raster) as src:
                raster_crs = src.crs
                rb = src.bounds
                minx, miny, maxx, maxy = rb.left, rb.bottom, rb.right, rb.top
            
            # Reproject geometry to raster CRS
            poly_gdf = farm_polygon1.to_crs(raster_crs)
            poly_coords = poly_gdf.iloc[0]['geometry']
            
            if hasattr(poly_coords, 'exterior'):
                img_h, img_w = INDEX_scaled.shape
                
                # Use geometry in the SAME CRS as the raster bounds
                coords = list(poly_coords.exterior.coords)
                scaled_coords = []
                for x, y in coords:
                    # Scale using RASTER bounds - this aligns polygon with raster extent
                    px = (x - minx) / (maxx - minx) * img_w
                    py = img_h - (y - miny) / (maxy - miny) * img_h
                    scaled_coords.append([px, py])
                
                # Create polygon patch for clipping
                polygon_patch = MplPolygon(scaled_coords, closed=True, transform=ax.transData)
                im.set_clip_path(polygon_patch)
            
            ax.axis('off')
            plt.savefig(png_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=150)
            plt.close(fig)
            tif_path = TIF_PATH+ AppNo+ '_'+str(st_date)[:10] + '_{}.tif'.format(s2_index)
            shape = INDEX.shape
            INDEX = INDEX.reshape(1,shape[0],shape[1])
            FarmVisionModelService.indices_to_tif(Clipped_raster, INDEX, tif_path)
            file_name = '{}/'.format(s2_index)+AppNo+ '_'+str(st_date)[:10] + '_{}.png'.format(s2_index)
            '''
            blob_client = BlobClient.from_connection_string(
                    conn_str='YOUR_CONNECTION_STRING',
                    container_name='indices',
                    blob_name=file_name)
            with open(png_path, "rb") as data:
                    blob_client.upload_blob(data,overwrite=True)
            '''
            #t3 = int(time.time())
            #print("TIME TAKEN for {} is {} Mins".format(st_date,round((t3-t2)/60,2)))
            return median_index,cloud_value,file_name#FarmVisionModelService.png2base64(png_path)         

    @staticmethod
    def process_INDEX(arguments):
            global all_data_df, farm_polygon, farm_polygon1, AppNo, s2_index,TEMP_PATH,TIF_PATH,PNG_PATH,JSON_PATH,signatures_path
            row = arguments[0]
            #print(f"process index for {row}")
            st_date = row
            en_date = arguments[1]#st_date + pd.DateOffset(5)
            filtered_data = all_data_df[(all_data_df["Date"] >= st_date) & (all_data_df["Date"] < en_date)]
            if len(filtered_data) == 0:
                return 1
            median_ndvi,cloudperct,png_ndvi = FarmVisionModelService.s2indexcompute(st_date,filtered_data,'NDVI')
            median_ndmi,cloudperct,png_ndmi = FarmVisionModelService.s2indexcompute(st_date,filtered_data,'NDMI')
            median_msavi,cloudperct,png_msavi = FarmVisionModelService.s2indexcompute(st_date,filtered_data,'MSAVI')
            median_ndre,cloudperct,png_ndre = FarmVisionModelService.s2indexcompute(st_date,filtered_data,'NDRE')
            median_gndvi,cloudperct,png_gndvi = FarmVisionModelService.s2indexcompute(st_date,filtered_data,'GNDVI')
            median_evi,cloudperct,png_evi = FarmVisionModelService.s2indexcompute(st_date,filtered_data,'EVI')
            #logger.info(f"process ndvi for {row} done")
            return cloudperct,median_ndvi,png_ndvi,median_ndmi,png_ndmi,median_msavi,png_msavi,median_ndre,png_ndre,median_gndvi,png_gndvi,median_evi,png_evi
    
    @staticmethod
    def plot_index(AppNo,dates,ndvi,ndmi,msavi,cloud):
          fig, ax1 = plt.subplots()
          mask = np.array(cloud)
          mask[mask<50] =1
          mask[mask>=50] = 0
          ndvi = list(np.array(ndvi)*mask*0.0001)
          ndmi = list(np.array(ndmi)*mask*0.0001)
          msavi = list(np.array(msavi)*mask*0.0001)
          ax1.plot(dates, ndvi, 'g-', label='NDVI')
          #ax1.plot(dates, ndmi, 'b-', label='NDMI')
          #ax1.plot(dates, msavi, 'r-', label='msavi')
          ax1.set_xlabel('Date')
          ax1.set_ylabel('NDVI', color='g')
          ax1.tick_params(axis='y', labelcolor='g')
          ax1.set_ylim(-0.1,0.7)
          ax2 = ax1.twinx()
          ax2.plot(dates, cloud,'o')
          ax2.set_ylabel('CLOUD', color='r')
          ax2.tick_params(axis='y', labelcolor='r')
          ax2.set_ylim(-5,105)
          ax1.legend(loc='upper left', bbox_to_anchor=(0, 0.8))
          plt.setp(ax1.get_xticklabels(),rotation = 80,ha="right", fontsize=10)
          plt.title('App {} Time Series of NDVI'.format(AppNo))
          plt.savefig(signatures_path+ "{}.png".format(AppNo),
                bbox_inches ="tight",orientation ='landscape')

    @staticmethod
    def calculateINDEXService(start_date,end_date,ApplNo,poly,temp):
            global all_data_df, farm_polygon,farm_polygon1,AppNo, s2_index,TEMP_PATH,TIF_PATH,PNG_PATH,JSON_PATH,signatures_path
            start_time = int(time.time())
            #logger.info("INFO :: calculating ndvi service")
            # start_date = '2024-11-01'
            # end_date  = '2025-03-10'
            time_series = start_date+'/'+end_date
            TEMP_PATH = temp
            TIF_PATH = TEMP_PATH + 'tif/'
            PNG_PATH = TEMP_PATH + 'png/'
            JSON_PATH = TEMP_PATH + 'json/'
            signatures_path = TEMP_PATH + 'signatures/'
            if not os.path.isdir(TEMP_PATH):
                 os.makedirs(TEMP_PATH,exist_ok = True)
            if not os.path.isdir(TIF_PATH):
                 os.makedirs(TIF_PATH,exist_ok = True)
            if not os.path.isdir(PNG_PATH):
                 os.makedirs(PNG_PATH,exist_ok = True)
            if not os.path.isdir(JSON_PATH):
                 os.makedirs(JSON_PATH,exist_ok = True)
            if not os.path.isdir(signatures_path):
                 os.makedirs(signatures_path,exist_ok = True)
            
            AppNo = str(ApplNo)
            farm_polygon1 = gpd.GeoDataFrame([poly],columns= ['geometry'],crs = 'epsg:4326')
            farm_polygon1 = farm_polygon1.to_crs(3857)
            farm_polygon = farm_polygon1.buffer(10)
            farm_polygon = gpd.GeoDataFrame(farm_polygon, columns=['geometry'])
            farm_polygon = farm_polygon.to_crs(4326)
            geometry = farm_polygon.iloc[0]['geometry']
            # query stac api
            client = Client.open(sentinal2_element84_url)
            sentinel_search = client.search(
                datetime=time_series,
                collections=[sentinal2_collection],
                intersects=geometry,
                query={"eo:cloud_cover": {"lt": 100}},
            )
            sentinel_items = sentinel_search.item_collection()
            series = [FarmVisionModelService.series_from(item) for item in sentinel_items]
            all_data_df = pd.DataFrame(series)
            all_data_df['Date'] = pd.to_datetime(all_data_df['Date'])
            date_range_5days = list(np.sort(all_data_df['Date'].unique()))
            print("date_range_5days {}".format(date_range_5days))
            date_range_5days.append(all_data_df.Date.max() + pd.DateOffset(1))
            #date_range_5days = pd.date_range(start=all_data_df.Date.min(),
            #                                 end=all_data_df.Date.max() + pd.DateOffset(5), freq='5D')
            date_range_5days_zip = list(zip(date_range_5days[:-1], date_range_5days[1:]))
            results = []
            #dates = list(date_range_5days.values)
            t1 = int(time.time())
            #print("TIME TAKEN for {} is {} Mins".format('fetching_satellite_data',round((t1-start_time)/60,2)))
            #logger.info("INFO :: Multithreading started")
            # Using sequential processing for Windows compatibility
            results = [FarmVisionModelService.process_INDEX(args) for args in date_range_5days_zip]
            results = list(filter(lambda x: x != 1, results))
            cloud = []
            ndvi = []
            png_ndvi = []
            ndmi= []
            png_ndmi = []
            msavi = []
            png_msavi = []
            ndre = []
            png_ndre = []
            gndvi = []
            png_gndvi = []
            evi = []
            png_evi = []
            for i in results:
                cloud.append(i[0])
                ndvi.append(i[1])
                png_ndvi.append(i[2])
                ndmi.append(i[3])
                png_ndmi.append(i[4])
                msavi.append(i[5])
                png_msavi.append(i[6])
                ndre.append(i[7])
                png_ndre.append(i[8])
                gndvi.append(i[9])
                png_gndvi.append(i[10])
                evi.append(i[11])
                png_evi.append(i[12])
            dates = [i.split('_')[-2] for i in png_ndvi]
            FarmVisionModelService.plot_index(AppNo,dates,ndvi,ndmi,msavi,cloud)
            final_dict = {
                         'cloud' : str(cloud),
                        'dates': str(dates),
                        'NDVI' : str(ndvi),
                        'NDMI' : str(ndmi),
                        'MSAVI' : str(msavi),
                        'NDRE' : str(ndre),
                        'GNDVI' : str(gndvi),
                        'EVI' : str(evi),
                        'png_ndvi' : png_ndvi,
                        'png_ndmi' : png_ndmi,
                        'png_msavi' : png_msavi,
                        'png_ndre' : png_ndre,
                        'png_gndvi' : png_gndvi,
                        'png_evi' : png_evi
                        }
            with open("{}S2_Indices_{}.json".format(JSON_PATH,AppNo), "w") as outfile: 
                json.dump(final_dict, outfile)
            
            # Clean up temp files (cross-platform)
            import glob
            for f in glob.glob(TEMP_PATH+'*.tif'):
                try:
                    os.remove(f)
                except:
                    pass
            end_time = int(time.time())
            print("TIME TAKEN for {} is {} Mins".format(AppNo,round((end_time-start_time)/60,2)))
            return final_dict



use_cpu = 16

if __name__ == '__main__':
    gdf = gpd.read_file('./polygon.shp')
    temp = './image_data/'
    start_date = '2025-05-01'
    end_date = '2025-05-10'
    for i,j in gdf.iterrows():
        FarmVisionModelService.calculateINDEXService(start_date, end_date, j.id, j.geometry, temp)


