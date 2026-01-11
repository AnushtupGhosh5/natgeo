"""
Tiling utilities for processing large orthophotos.
Handles splitting images into tiles and stitching predictions back.
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from typing import Tuple, List, Generator
from dataclasses import dataclass


@dataclass
class TileInfo:
    """Information about a single tile."""
    x: int  # Column offset
    y: int  # Row offset
    width: int  # Tile width
    height: int  # Tile height
    pad_right: int  # Padding added to right
    pad_bottom: int  # Padding added to bottom


def calculate_tile_grid(
    image_width: int,
    image_height: int,
    tile_size: int = 512,
    overlap: int = 0
) -> List[TileInfo]:
    """
    Calculate the grid of tiles needed to cover the entire image.
    
    Args:
        image_width: Width of the input image
        image_height: Height of the input image
        tile_size: Size of each tile (default 512)
        overlap: Overlap between adjacent tiles (default 0)
    
    Returns:
        List of TileInfo objects describing each tile
    """
    stride = tile_size - overlap
    tiles = []
    
    y = 0
    while y < image_height:
        x = 0
        while x < image_width:
            # Calculate actual tile dimensions (may be smaller at edges)
            actual_width = min(tile_size, image_width - x)
            actual_height = min(tile_size, image_height - y)
            
            # Calculate padding needed
            pad_right = tile_size - actual_width
            pad_bottom = tile_size - actual_height
            
            tiles.append(TileInfo(
                x=x,
                y=y,
                width=actual_width,
                height=actual_height,
                pad_right=pad_right,
                pad_bottom=pad_bottom
            ))
            
            x += stride
            if x >= image_width:
                break
        
        y += stride
        if y >= image_height:
            break
    
    return tiles


def extract_tiles_from_array(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 0
) -> Tuple[List[np.ndarray], List[TileInfo]]:
    """
    Extract tiles from a numpy array image.
    
    Args:
        image: Input image array of shape (C, H, W) or (H, W, C)
        tile_size: Size of each tile
        overlap: Overlap between tiles
    
    Returns:
        Tuple of (list of tile arrays, list of TileInfo)
    """
    # Ensure image is (C, H, W) format
    if image.ndim == 3 and image.shape[2] in [1, 3, 4]:
        image = np.transpose(image, (2, 0, 1))
    elif image.ndim == 2:
        image = image[np.newaxis, :, :]
    
    channels, height, width = image.shape
    tiles_info = calculate_tile_grid(width, height, tile_size, overlap)
    tiles = []
    
    for info in tiles_info:
        # Extract tile region
        tile = image[:, info.y:info.y + info.height, info.x:info.x + info.width]
        
        # Pad if necessary
        if info.pad_right > 0 or info.pad_bottom > 0:
            tile = np.pad(
                tile,
                ((0, 0), (0, info.pad_bottom), (0, info.pad_right)),
                mode='reflect'
            )
        
        tiles.append(tile)
    
    return tiles, tiles_info


def extract_tiles_from_rasterio(
    raster_path: str,
    tile_size: int = 512,
    overlap: int = 0,
    max_bands: int = 3
) -> Generator[Tuple[np.ndarray, TileInfo], None, None]:
    """
    Generator that yields tiles from a rasterio dataset.
    Memory-efficient for large images.
    
    Args:
        raster_path: Path to the raster file
        tile_size: Size of each tile
        overlap: Overlap between tiles
        max_bands: Maximum number of bands to read (default 3 for RGB)
    
    Yields:
        Tuple of (tile array, TileInfo)
    """
    with rasterio.open(raster_path) as src:
        width, height = src.width, src.height
        tiles_info = calculate_tile_grid(width, height, tile_size, overlap)
        
        # Determine bands to read
        bands_to_read = list(range(1, min(src.count, max_bands) + 1))
        
        for info in tiles_info:
            window = Window(info.x, info.y, info.width, info.height)
            tile = src.read(bands_to_read, window=window)
            
            # Pad if necessary
            if info.pad_right > 0 or info.pad_bottom > 0:
                tile = np.pad(
                    tile,
                    ((0, 0), (0, info.pad_bottom), (0, info.pad_right)),
                    mode='reflect'
                )
            
            yield tile, info


def stitch_tiles(
    tiles: List[np.ndarray],
    tiles_info: List[TileInfo],
    output_shape: Tuple[int, int],
    overlap: int = 0
) -> np.ndarray:
    """
    Stitch predicted tiles back into a full-resolution mask.
    
    Args:
        tiles: List of predicted tile arrays (each of shape (H, W) or (1, H, W))
        tiles_info: List of TileInfo objects
        output_shape: (height, width) of the output image
        overlap: Overlap that was used during tiling
    
    Returns:
        Stitched mask of shape (H, W)
    """
    height, width = output_shape
    output = np.zeros((height, width), dtype=np.float32)
    
    # For overlapping regions, we use averaging
    if overlap > 0:
        count = np.zeros((height, width), dtype=np.float32)
    
    for tile, info in zip(tiles, tiles_info):
        # Remove batch/channel dimension if present
        if tile.ndim == 3:
            tile = tile.squeeze()
        elif tile.ndim == 4:
            tile = tile.squeeze()
        
        # Get the valid (non-padded) region of the tile
        valid_tile = tile[:info.height, :info.width]
        
        # Calculate output region
        y_start = info.y
        y_end = info.y + info.height
        x_start = info.x
        x_end = info.x + info.width
        
        if overlap > 0:
            output[y_start:y_end, x_start:x_end] += valid_tile
            count[y_start:y_end, x_start:x_end] += 1
        else:
            output[y_start:y_end, x_start:x_end] = valid_tile
    
    if overlap > 0:
        # Average overlapping regions
        count = np.maximum(count, 1)  # Avoid division by zero
        output = output / count
    
    return output


def load_orthophoto_for_preview(
    raster_path: str,
    max_size: int = 1024
) -> Tuple[np.ndarray, dict]:
    """
    Load a downsampled version of orthophoto for preview.
    
    Args:
        raster_path: Path to the raster file
        max_size: Maximum dimension for the preview
    
    Returns:
        Tuple of (preview image array (H, W, C), metadata dict)
    """
    with rasterio.open(raster_path) as src:
        # Calculate downsampling factor
        scale = min(max_size / src.width, max_size / src.height, 1.0)
        
        out_width = int(src.width * scale)
        out_height = int(src.height * scale)
        
        # Read downsampled
        bands_to_read = list(range(1, min(src.count, 3) + 1))
        img = src.read(
            bands_to_read,
            out_shape=(len(bands_to_read), out_height, out_width),
            resampling=rasterio.enums.Resampling.bilinear
        )
        
        # Transpose to (H, W, C)
        img = np.transpose(img, (1, 2, 0))
        
        metadata = {
            'width': src.width,
            'height': src.height,
            'crs': str(src.crs) if src.crs else None,
            'bounds': src.bounds,
            'transform': src.transform,
            'count': src.count,
            'dtype': str(src.dtypes[0])
        }
        
        return img, metadata
