
import os
import rasterio
import numpy as np
import cv2
import glob
import random
import matplotlib.pyplot as plt
from rasterio.windows import Window
from tqdm import tqdm
from PIL import Image

# 1. Configuration
IMG_PATH = "../data/images/28996_NADALA_ORTHO.tif"
MASK_PATH = "../data/masks/utility_mask_full.tif"
OUTPUT_DIR = "../data/tiles"
IMG_OUT_DIR = os.path.join(OUTPUT_DIR, "images")
MASK_OUT_DIR = os.path.join(OUTPUT_DIR, "masks")
TILE_SIZE = 1024
STRIDE = 1024

os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(MASK_OUT_DIR, exist_ok=True)

def generate_tiles(img_path, mask_path):
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print("Input file(s) missing.")
        return

    print("Generating tiles...")
    with rasterio.open(img_path) as src_img, rasterio.open(mask_path) as src_mask:
        width, height = src_img.width, src_img.height
        
        # Calculate grid
        x_steps = list(range(0, width, STRIDE))
        y_steps = list(range(0, height, STRIDE))
        
        count = 0
        saved_count = 0
        
        pbar = tqdm(total=len(x_steps)*len(y_steps), desc="Tiling")
        
        for row in y_steps:
            for col in x_steps:
                # Window definition
                # We crop only full tiles or handle edges if needed (here we clip edges)
                w_width = min(TILE_SIZE, width - col)
                w_height = min(TILE_SIZE, height - row)
                
                window = Window(col, row, w_width, w_height)
                
                # Check mask content first (fast check)
                mask_data = src_mask.read(1, window=window)
                
                # Skip if empty (all zeros)
                if not np.any(mask_data):
                    pbar.update(1)
                    continue
                
                # If we are at edge and size is smaller than TILE_SIZE, pad it
                # Or simply skip edges for strict sizing (easier for standard UNet)
                if w_width < TILE_SIZE or w_height < TILE_SIZE:
                    # Pad to 1024x1024
                    pad_mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=mask_data.dtype)
                    pad_mask[:w_height, :w_width] = mask_data
                    mask_final = pad_mask
                    
                    img_data = src_img.read(window=window) # (C, H, W)
                    pad_img = np.zeros((src_img.count, TILE_SIZE, TILE_SIZE), dtype=img_data.dtype)
                    pad_img[:, :w_height, :w_width] = img_data
                    img_final = pad_img
                else:
                    mask_final = mask_data
                    img_final = src_img.read(window=window)

                # Save
                tile_name = f"tile_{col}_{row}.png" # Use PNG for lossless
                
                # Transpose image to (H,W,C) for Pillow
                img_array = img_final.transpose(1, 2, 0)
                
                # Save Image (RGB)
                Image.fromarray(img_array).save(os.path.join(IMG_OUT_DIR, tile_name))
                
                # Save Mask (Grayscale)
                # Convert 1s to 255 for visualization/compatibility, or keep 1?
                # Usually standard masks are 0-255. Let's save as 0-255.
                Image.fromarray(mask_final * 255).save(os.path.join(MASK_OUT_DIR, tile_name))
                
                saved_count += 1
                pbar.update(1)
                
    print(f"\nDone! Processed {len(x_steps)*len(y_steps)} windows. Saved {saved_count} tiles containing utilities.")

def visualize_samples(n=5):
    print(f"\nVisualizing {n} random samples...")
    images = glob.glob(os.path.join(IMG_OUT_DIR, "*.png"))
    if not images:
        print("No tiles found.")
        return
    
    samples = random.sample(images, min(n, len(images)))
    
    for img_p in samples:
        filename = os.path.basename(img_p)
        mask_p = os.path.join(MASK_OUT_DIR, filename)
        
        img = Image.open(img_p)
        mask = Image.open(mask_p)
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        ax[0].imshow(img)
        ax[0].set_title("Input Image")
        ax[0].axis('off')
        
        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title("Ground Truth Mask")
        ax[1].axis('off')
        
        # Overlay
        img_np = np.array(img)
        mask_np = np.array(mask)
        overlay = img_np.copy()
        overlay[mask_np > 0] = [255, 0, 0] # Red overlay
        
        ax[2].imshow(overlay)
        ax[2].set_title("Overlay")
        ax[2].axis('off')
        
        plt.show()

# Execute
generate_tiles(IMG_PATH, MASK_PATH)
visualize_samples()
