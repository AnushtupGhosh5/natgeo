"""
Post-processing utilities for segmentation masks.
Includes morphological operations, thresholding, and cleanup.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PostProcessConfig:
    """Configuration for post-processing."""
    apply_morphology: bool = True
    kernel_size: int = 3
    min_area: int = 100  # Minimum connected component area
    fill_holes: bool = True
    smooth_boundaries: bool = True


def apply_morphological_ops(
    mask: np.ndarray,
    kernel_size: int = 3,
    operation: str = 'close'
) -> np.ndarray:
    """
    Apply morphological operations to clean up the mask.
    
    Args:
        mask: Binary mask (0 or 1)
        kernel_size: Size of the morphological kernel
        operation: 'open', 'close', or 'both'
    
    Returns:
        Processed mask
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (kernel_size, kernel_size)
    )
    
    # Ensure mask is uint8
    mask = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
    
    if operation == 'open':
        result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif operation == 'both':
        result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    else:
        result = mask
    
    return (result > 127).astype(np.uint8)


def remove_small_components(
    mask: np.ndarray,
    min_area: int = 100
) -> np.ndarray:
    """
    Remove small connected components from the mask.
    
    Args:
        mask: Binary mask
        min_area: Minimum area (in pixels) to keep
    
    Returns:
        Cleaned mask
    """
    # Ensure mask is uint8
    mask = mask.astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    
    # Create output mask
    output = np.zeros_like(mask)
    
    # Keep only components larger than min_area
    for i in range(1, num_labels):  # Skip background (0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            output[labels == i] = 1
    
    return output


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in the binary mask.
    
    Args:
        mask: Binary mask
    
    Returns:
        Mask with holes filled
    """
    # Ensure mask is uint8
    mask = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
    
    # Flood fill from the edges
    h, w = mask.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    inverted = cv2.bitwise_not(mask)
    cv2.floodFill(inverted, flood_mask, (0, 0), 255)
    
    # Invert the flood-filled inverse to get holes
    holes = cv2.bitwise_not(inverted)
    
    # Combine with original mask
    filled = cv2.bitwise_or(mask, holes)
    
    return (filled > 127).astype(np.uint8)


def smooth_boundaries(
    mask: np.ndarray,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Smooth the boundaries of the mask using Gaussian blur.
    
    Args:
        mask: Binary mask
        sigma: Gaussian blur sigma
    
    Returns:
        Mask with smoothed boundaries
    """
    # Ensure mask is float
    mask_float = mask.astype(np.float32)
    
    # Apply Gaussian blur
    kernel_size = int(sigma * 4) | 1  # Ensure odd
    blurred = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), sigma)
    
    # Re-threshold
    return (blurred > 0.5).astype(np.uint8)


def post_process_mask(
    mask: np.ndarray,
    config: Optional[PostProcessConfig] = None
) -> np.ndarray:
    """
    Apply full post-processing pipeline to a mask.
    
    Args:
        mask: Binary mask (values 0 or 1)
        config: Post-processing configuration
    
    Returns:
        Post-processed mask
    """
    if config is None:
        config = PostProcessConfig()
    
    result = mask.copy()
    
    # Ensure binary
    if result.max() > 1:
        result = (result > 127).astype(np.uint8)
    else:
        result = result.astype(np.uint8)
    
    if config.apply_morphology:
        # Close then open to remove noise and fill gaps
        result = apply_morphological_ops(
            result, config.kernel_size, 'both'
        )
    
    if config.fill_holes:
        result = fill_holes(result)
    
    if config.min_area > 0:
        result = remove_small_components(result, config.min_area)
    
    if config.smooth_boundaries:
        result = smooth_boundaries(result)
    
    return result


def get_feature_color(feature_type: str) -> Tuple[int, int, int]:
    """
    Get the RGB color for a feature type.
    
    Args:
        feature_type: Type of feature
    
    Returns:
        RGB tuple (0-255)
    """
    colors = {
        'buildings': (255, 120, 50),      # Orange
        'roads': (64, 64, 64),             # Dark gray
        'water_bodies': (30, 144, 255),    # Dodger blue
        'utilities': (34, 139, 34),        # Forest green
        'railways': (139, 69, 19),         # Saddle brown
    }
    return colors.get(feature_type, (255, 0, 0))


def create_colored_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    feature_type: str,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create a colored overlay of the mask on the original image.
    
    Args:
        image: Original RGB image (H, W, C)
        mask: Binary mask (H, W)
        feature_type: Type of feature for color selection
        alpha: Overlay transparency (0-1)
    
    Returns:
        Overlay image (H, W, C)
    """
    # Ensure image is RGB and uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        elif image.max() > 255:
            image = (image / 256).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Handle different image shapes
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.shape[0] in [1, 3, 4]:  # (C, H, W) format
        image = np.transpose(image, (1, 2, 0))
    
    # Ensure 3 channels
    if image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    
    # Get feature color
    color = get_feature_color(feature_type)
    
    # Create overlay
    overlay = image.copy()
    mask_bool = mask.astype(bool)
    
    # Blend color with original where mask is positive
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            (1 - alpha) * image[:, :, c] + alpha * color[c],
            image[:, :, c]
        )
    
    return overlay.astype(np.uint8)


def create_mask_visualization(
    mask: np.ndarray,
    feature_type: str
) -> np.ndarray:
    """
    Create a colored visualization of the mask.
    
    Args:
        mask: Binary mask (H, W)
        feature_type: Type of feature for color selection
    
    Returns:
        Colored mask image (H, W, C)
    """
    color = get_feature_color(feature_type)
    
    # Create RGB image
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    mask_bool = mask.astype(bool)
    for c in range(3):
        colored[:, :, c] = np.where(mask_bool, color[c], 0)
    
    return colored
