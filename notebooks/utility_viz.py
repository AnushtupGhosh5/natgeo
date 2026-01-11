
import os
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.plot import show
import numpy as np

# 1. Paths configuration
IMG_PATH = "/home/anushtup-ghosh/Documents/Projects/natgeo/data/images/28996_NADALA_ORTHO.tif"
SHP_UTILITY_POINT = "/home/anushtup-ghosh/Documents/Projects/natgeo/data/shp/Utility.shp"
SHP_UTILITY_POLY = "/home/anushtup-ghosh/Documents/Projects/natgeo/data/shp/Utility_Poly_.shp" 
SHP_WATERBODY = "/home/anushtup-ghosh/Documents/Projects/natgeo/data/shp/Waterbody_Point.shp"

def visualize_utility_overlay(img_path, shp_paths):
    # Check if image exists
    if not os.path.exists(img_path):
        # Fallback to verify directory if specific file missing
        print(f"Image not found at {img_path}")
        return

    # Open Orthophoto (Memory Efficiently)
    try:
        with rasterio.open(img_path) as src:
            print(f"Image CRS: {src.crs}")
            print(f"Image bounds: {src.bounds}")
            
            # Read a downsampled overview for visualization (e.g., 1/16th scale)
            # This prevents crashing on large TIFFs
            out_shape = (
                src.count,
                int(src.height / 16),
                int(src.width / 16)
            )
            img = src.read(out_shape=out_shape, resampling=rasterio.enums.Resampling.bilinear)
            
            # Update transform for the downsampled image so coordinates align
            transform = src.transform * src.transform.scale(
                (src.width / out_shape[-1]),
                (src.height / out_shape[-2])
            )
            
            img_crs = src.crs

            # Prepare plotting
            fig, ax = plt.subplots(figsize=(12, 12))
            
            # Helper to normalize for display
            def normalize(array):
                array_min, array_max = array.min(), array.max()
                return (array - array_min) / (array_max - array_min)

            # Display RGB Image
            # show() takes the transform to place it correctly in coordinates
            show(img, transform=transform, ax=ax, title="Utility Overlay Verification")
            
            # Colors for different shapefiles
            colors = ['red', 'blue', 'cyan']
            
            # Load and Overlay Shapefiles
            for i, shp_path in enumerate(shp_paths):
                if os.path.exists(shp_path):
                    print(f"\nProcessing {os.path.basename(shp_path)}...")
                    gdf = gpd.read_file(shp_path)
                    print(f"Original SHP CRS: {gdf.crs}")
                    
                    # Reproject if needed
                    if gdf.crs != img_crs:
                        print(f"Reprojecting to {img_crs}...")
                        gdf = gdf.to_crs(img_crs)
                    
                    # Filter data within image bounds to speed up plotting
                    # (Optional but good for huge shapefiles)
                    minx, miny, maxx, maxy = src.bounds
                    gdf = gdf.cx[minx:maxx, miny:maxy]
                    
                    if not gdf.empty:
                        gdf.plot(ax=ax, color=colors[i % len(colors)], markersize=5, 
                               label=os.path.basename(shp_path), alpha=0.7, edgecolor='black')
                        print("Overlay added.")
                    else:
                        print("No features found within image bounds.")
                else:
                    print(f"File not found: {shp_path}")
            
            plt.legend()
            plt.show()
            
    except Exception as e:
        print(f"Error occurred: {e}")

# Run Visualization
visualize_utility_overlay(IMG_PATH, [SHP_UTILITY_POINT, SHP_UTILITY_POLY, SHP_WATERBODY])
