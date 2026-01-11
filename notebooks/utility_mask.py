
import os
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import Point, LineString, Polygon
from tqdm import tqdm

# 1. Configuration
IMG_PATH = "../data/images/28996_NADALA_ORTHO.tif"
SHP_PATHS = [
    "../data/shp/Utility.shp",      # Points
    "../data/shp/Utility_Poly_.shp" # Polygons
]
OUTPUT_MASK_PATH = "../data/masks/utility_mask_full.tif"

# Create output dir
os.makedirs(os.path.dirname(OUTPUT_MASK_PATH), exist_ok=True)

def generate_utility_mask(img_path, shp_paths, output_path):
    print(f"Generating mask for: {img_path}")
    
    with rasterio.open(img_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
        out_profile = src.profile.copy()
        
    print(f"Dimensions: {width}x{height}")
    
    # 2. Collect all geometries
    all_geometries = []
    
    for shp_path in shp_paths:
        if not os.path.exists(shp_path):
            print(f"Skipping missing file: {shp_path}")
            continue
            
        print(f"Loading {os.path.basename(shp_path)}...")
        gdf = gpd.read_file(shp_path)
        
        # Reproject if necessary
        if gdf.crs != crs:
            print(f"Reprojecting from {gdf.crs} to {crs}...")
            gdf = gdf.to_crs(crs)
            
        # Add to list
        # For points, buffer them slightly to make them visible in raster (e.g., 1 meter radius)
        # Adjust 'buffer_size' based on resolution (check src.res)
        buffer_size = src.res[0] * 3 # 3 pixel radius
        
        for geom in tqdm(gdf.geometry, desc=f"Processing {os.path.basename(shp_path)}"):
            if geom.is_empty:
                continue
                
            if isinstance(geom, Point):
                # Buffer points to circles
                all_geometries.append(geom.buffer(buffer_size))
            elif isinstance(geom, LineString):
                # Buffer lines to have thickness
                all_geometries.append(geom.buffer(buffer_size / 2))
            else:
                # Polygons added as is
                all_geometries.append(geom)

    if not all_geometries:
        print("No valid geometries found to rasterize.")
        return

    print(f"Rasterizing {len(all_geometries)} geometries...")
    
    # 3. Rasterize (Memory Efficient)
    # If image is too large for RAM, we might need a tiled approach.
    # For now, we assume user has 16GB+ RAM or images are <4GB.
    
    # Prepare iterable (geom, value)
    shapes = ((g, 255) for g in all_geometries)
    
    # Rasterize
    mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype='uint8'
    )
    
    # 4. Save
    out_profile.update(dtype=rasterio.uint8, count=1, compress='lzw', nodata=0)
    
    with rasterio.open(output_path, 'w', **out_profile) as dst:
        dst.write(mask, 1)
        
    print(f"Saved binary utility mask to: {output_path}")

# Run
generate_utility_mask(IMG_PATH, SHP_PATHS, OUTPUT_MASK_PATH)
