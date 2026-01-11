"""
Inference engine for running model predictions on orthophotos.
Handles batched inference, GPU/CPU selection, and tile processing.

SUPPORTS LARGE FILES (1-2GB+) via streaming tile reading from rasterio.
"""

import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Generator
from dataclasses import dataclass
import gc

from .tiling import TileInfo, calculate_tile_grid, stitch_tiles


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    tile_size: int = 512
    batch_size: int = 8
    overlap: int = 0
    threshold: float = 0.5
    device: Optional[str] = None  # Auto-detect if None


def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Get the best available device for inference.
    
    Args:
        preferred: Preferred device ('cuda', 'cpu', or None for auto)
    
    Returns:
        torch.device object
    """
    if preferred:
        return torch.device(preferred)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def normalize_tile(tile: np.ndarray) -> np.ndarray:
    """
    Normalize a tile for model input.
    
    Args:
        tile: Input tile of shape (C, H, W) with values 0-255
    
    Returns:
        Normalized tile with values 0-1
    """
    tile = tile.astype(np.float32)
    
    # Handle different bit depths
    if tile.max() > 1.0:
        if tile.max() > 255:
            tile = tile / 65535.0  # 16-bit
        else:
            tile = tile / 255.0  # 8-bit
    
    return tile


def prepare_batch(tiles: List[np.ndarray], device: torch.device) -> torch.Tensor:
    """
    Prepare a batch of tiles for inference.
    
    Args:
        tiles: List of tile arrays (C, H, W)
        device: Target device
    
    Returns:
        Batch tensor of shape (N, C, H, W)
    """
    # Normalize tiles
    normalized = [normalize_tile(tile) for tile in tiles]
    
    # Stack into batch
    batch = np.stack(normalized, axis=0)
    
    # Convert to tensor
    tensor = torch.from_numpy(batch).to(device)
    
    return tensor


def read_tile_from_rasterio(
    src: rasterio.DatasetReader,
    info: TileInfo,
    bands_to_read: List[int],
    tile_size: int
) -> np.ndarray:
    """
    Read a single tile from a rasterio dataset.
    
    Args:
        src: Open rasterio dataset
        info: TileInfo describing the tile location
        bands_to_read: List of band indices (1-indexed)
        tile_size: Target tile size for padding
    
    Returns:
        Tile array of shape (C, tile_size, tile_size)
    """
    window = Window(info.x, info.y, info.width, info.height)
    tile = src.read(bands_to_read, window=window)
    
    # Pad if necessary to reach tile_size
    if info.pad_right > 0 or info.pad_bottom > 0:
        tile = np.pad(
            tile,
            ((0, 0), (0, info.pad_bottom), (0, info.pad_right)),
            mode='reflect'
        )
    
    return tile


def streaming_tile_generator(
    raster_path: str,
    tiles_info: List[TileInfo],
    tile_size: int,
    max_bands: int = 3
) -> Generator[Tuple[np.ndarray, TileInfo], None, None]:
    """
    Generator that yields tiles one by one from a raster file.
    Memory efficient - only one tile in memory at a time.
    
    Args:
        raster_path: Path to the raster file
        tiles_info: List of TileInfo objects
        tile_size: Target tile size
        max_bands: Maximum bands to read
    
    Yields:
        Tuple of (tile array, TileInfo)
    """
    with rasterio.open(raster_path) as src:
        bands_to_read = list(range(1, min(src.count, max_bands) + 1))
        
        for info in tiles_info:
            tile = read_tile_from_rasterio(src, info, bands_to_read, tile_size)
            yield tile, info


def run_streaming_inference(
    model: torch.nn.Module,
    raster_path: str,
    config: InferenceConfig,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a large raster file using streaming tile reading.
    
    This is the MEMORY-EFFICIENT version for large (1-2GB+) GeoTIFFs.
    Tiles are read one batch at a time directly from disk.
    
    Args:
        model: Loaded PyTorch model
        raster_path: Path to the GeoTIFF file
        config: Inference configuration
        progress_callback: Callback for progress updates (progress, stage_name)
    
    Returns:
        Tuple of (probability_mask, binary_mask)
    """
    def update_progress(p, stage):
        if progress_callback:
            progress_callback(p, stage)
    
    device = get_device(config.device)
    model = model.to(device)
    model.eval()
    
    # Get image dimensions
    with rasterio.open(raster_path) as src:
        width, height = src.width, src.height
        n_bands = min(src.count, 3)
    
    update_progress(0.0, f"Image size: {width:,} x {height:,} pixels")
    
    # Calculate tile grid
    tiles_info = calculate_tile_grid(width, height, config.tile_size, config.overlap)
    total_tiles = len(tiles_info)
    
    update_progress(0.02, f"Processing {total_tiles:,} tiles...")
    
    # Pre-allocate output mask (only store the mask, not the image)
    # This is much smaller than the full image
    prob_mask = np.zeros((height, width), dtype=np.float32)
    
    # For overlap handling
    if config.overlap > 0:
        count_mask = np.zeros((height, width), dtype=np.float32)
    
    # Process tiles in batches with streaming
    processed_tiles = 0
    batch_tiles = []
    batch_info = []
    
    with rasterio.open(raster_path) as src:
        bands_to_read = list(range(1, n_bands + 1))
        
        with torch.no_grad():
            for info in tiles_info:
                # Read tile from disk
                tile = read_tile_from_rasterio(src, info, bands_to_read, config.tile_size)
                batch_tiles.append(tile)
                batch_info.append(info)
                
                # Process batch when full or at end
                if len(batch_tiles) >= config.batch_size or info == tiles_info[-1]:
                    # Prepare batch
                    batch_tensor = prepare_batch(batch_tiles, device)
                    
                    # Forward pass
                    outputs = model(batch_tensor)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    
                    # Place predictions in output mask
                    for j, (pred, tile_info) in enumerate(zip(probs, batch_info)):
                        pred_2d = pred.squeeze()
                        
                        # Get valid region (without padding)
                        valid_pred = pred_2d[:tile_info.height, :tile_info.width]
                        
                        y_start = tile_info.y
                        y_end = tile_info.y + tile_info.height
                        x_start = tile_info.x
                        x_end = tile_info.x + tile_info.width
                        
                        if config.overlap > 0:
                            prob_mask[y_start:y_end, x_start:x_end] += valid_pred
                            count_mask[y_start:y_end, x_start:x_end] += 1
                        else:
                            prob_mask[y_start:y_end, x_start:x_end] = valid_pred
                    
                    # Update progress
                    processed_tiles += len(batch_tiles)
                    progress = 0.05 + (processed_tiles / total_tiles) * 0.85
                    update_progress(progress, f"Processing tiles... {processed_tiles}/{total_tiles}")
                    
                    # Clear batch
                    batch_tiles = []
                    batch_info = []
                    
                    # Clear GPU memory periodically
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
    
    # Handle overlapping regions
    if config.overlap > 0:
        count_mask = np.maximum(count_mask, 1)
        prob_mask = prob_mask / count_mask
    
    # Apply threshold
    update_progress(0.95, "Applying threshold...")
    binary_mask = (prob_mask >= config.threshold).astype(np.uint8)
    
    # Clean up
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    update_progress(1.0, "Complete!")
    
    return prob_mask, binary_mask


def run_batched_inference(
    model: torch.nn.Module,
    tiles: List[np.ndarray],
    tiles_info: List[TileInfo],
    config: InferenceConfig,
    progress_callback: Optional[Callable[[float], None]] = None
) -> List[np.ndarray]:
    """
    Run batched inference on tiles (for small images already in memory).
    
    Args:
        model: Loaded PyTorch model
        tiles: List of tile arrays
        tiles_info: List of TileInfo objects
        config: Inference configuration
        progress_callback: Optional callback for progress updates (0.0 to 1.0)
    
    Returns:
        List of predicted mask tiles
    """
    device = get_device(config.device)
    model = model.to(device)
    model.eval()
    
    predictions = []
    total_tiles = len(tiles)
    
    with torch.no_grad():
        for i in range(0, total_tiles, config.batch_size):
            batch_tiles = tiles[i:i + config.batch_size]
            batch_tensor = prepare_batch(batch_tiles, device)
            
            # Forward pass
            outputs = model(batch_tensor)
            
            # Apply sigmoid and convert to numpy
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # Store predictions
            for j in range(probs.shape[0]):
                predictions.append(probs[j])
            
            # Update progress
            if progress_callback:
                progress = min((i + config.batch_size) / total_tiles, 1.0)
                progress_callback(progress)
            
            # Clear GPU memory periodically
            if device.type == 'cuda' and (i + config.batch_size) % (config.batch_size * 4) == 0:
                torch.cuda.empty_cache()
    
    return predictions


def run_inference_pipeline(
    model: torch.nn.Module,
    image: np.ndarray,
    config: InferenceConfig,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the complete inference pipeline on an image array.
    
    NOTE: For large files (>500MB), use run_streaming_inference() instead.
    
    Args:
        model: Loaded PyTorch model
        image: Input image array of shape (C, H, W)
        config: Inference configuration
        progress_callback: Callback for progress updates (progress, stage_name)
    
    Returns:
        Tuple of (probability_mask, binary_mask)
    """
    def update_progress(p, stage):
        if progress_callback:
            progress_callback(p, stage)
    
    # Ensure correct format
    if image.ndim == 3 and image.shape[2] in [1, 3, 4]:
        image = np.transpose(image, (2, 0, 1))
    
    channels, height, width = image.shape
    
    # Step 1: Calculate tile grid
    update_progress(0.0, "Calculating tile grid...")
    tiles_info = calculate_tile_grid(width, height, config.tile_size, config.overlap)
    
    # Step 2: Extract tiles
    update_progress(0.05, f"Extracting {len(tiles_info)} tiles...")
    tiles = []
    for info in tiles_info:
        tile = image[:, info.y:info.y + info.height, info.x:info.x + info.width]
        
        # Pad if necessary
        if info.pad_right > 0 or info.pad_bottom > 0:
            tile = np.pad(
                tile,
                ((0, 0), (0, info.pad_bottom), (0, info.pad_right)),
                mode='reflect'
            )
        tiles.append(tile)
    
    # Step 3: Run batched inference
    update_progress(0.1, "Running model inference...")
    
    def inference_progress(p):
        # Map inference progress to 10%-90%
        overall = 0.1 + p * 0.8
        update_progress(overall, f"Processing tiles... ({int(p*100)}%)")
    
    predictions = run_batched_inference(
        model, tiles, tiles_info, config, inference_progress
    )
    
    # Step 4: Stitch predictions
    update_progress(0.9, "Stitching predictions...")
    prob_mask = stitch_tiles(predictions, tiles_info, (height, width), config.overlap)
    
    # Step 5: Apply threshold
    update_progress(0.95, "Applying threshold...")
    binary_mask = (prob_mask >= config.threshold).astype(np.uint8)
    
    update_progress(1.0, "Complete!")
    
    return prob_mask, binary_mask


def load_model(model_path: str, device: Optional[str] = None) -> torch.nn.Module:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the model weights (.pth file)
        device: Target device (auto-detect if None)
    
    Returns:
        Loaded model ready for inference
    """
    import segmentation_models_pytorch as smp
    
    device = get_device(device)
    
    # Initialize model - the saved weights are from smp.Unet with resnet34 encoder
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # We'll load our own weights
        in_channels=3,
        classes=1,
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    return model
