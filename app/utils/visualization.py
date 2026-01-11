"""
Visualization utilities for displaying results in Streamlit.
"""

import numpy as np
import io
from PIL import Image
from typing import Tuple, Optional
import cv2


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image for display.
    
    Args:
        image: Input image array
    
    Returns:
        Normalized uint8 image suitable for display
    """
    if image.dtype == np.uint8:
        return image
    
    if image.max() <= 1.0:
        return (image * 255).astype(np.uint8)
    elif image.max() > 255:
        # 16-bit or higher
        return ((image / image.max()) * 255).astype(np.uint8)
    else:
        return image.astype(np.uint8)


def ensure_hwc_format(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is in (H, W, C) format.
    
    Args:
        image: Input image
    
    Returns:
        Image in (H, W, C) format
    """
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    elif image.shape[0] in [1, 3, 4] and image.ndim == 3:
        return np.transpose(image, (1, 2, 0))
    return image


def resize_for_display(
    image: np.ndarray,
    max_width: int = 800,
    max_height: int = 800
) -> np.ndarray:
    """
    Resize image for display while maintaining aspect ratio.
    
    Args:
        image: Input image (H, W, C)
        max_width: Maximum width
        max_height: Maximum height
    
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    # Calculate scale factor
    scale_w = max_width / w if w > max_width else 1.0
    scale_h = max_height / h if h > max_height else 1.0
    scale = min(scale_w, scale_h)
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image


def create_side_by_side(
    images: list,
    titles: Optional[list] = None,
    gap: int = 10
) -> np.ndarray:
    """
    Create a side-by-side visualization of multiple images.
    
    Args:
        images: List of images (H, W, C)
        titles: Optional list of titles
        gap: Gap between images in pixels
    
    Returns:
        Combined image
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Normalize all images
    processed = []
    for img in images:
        img = ensure_hwc_format(img)
        img = normalize_for_display(img)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        processed.append(img)
    
    # Find max height
    max_h = max(img.shape[0] for img in processed)
    
    # Resize all to same height
    resized = []
    for img in processed:
        if img.shape[0] != max_h:
            scale = max_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_LINEAR)
        resized.append(img)
    
    # Calculate total width
    total_w = sum(img.shape[1] for img in resized) + gap * (len(resized) - 1)
    
    # Create output
    output = np.ones((max_h, total_w, 3), dtype=np.uint8) * 255
    
    # Place images
    x = 0
    for img in resized:
        w = img.shape[1]
        output[:, x:x+w, :] = img
        x += w + gap
    
    return output


def image_to_bytes(image: np.ndarray, format: str = 'PNG') -> bytes:
    """
    Convert numpy image to bytes for download.
    
    Args:
        image: Input image
        format: Output format ('PNG', 'JPEG', 'TIFF')
    
    Returns:
        Image bytes
    """
    image = normalize_for_display(image)
    image = ensure_hwc_format(image)
    
    if image.shape[2] > 3:
        image = image[:, :, :3]
    
    pil_image = Image.fromarray(image)
    
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)
    
    return buffer.getvalue()


def mask_to_bytes(mask: np.ndarray, format: str = 'PNG') -> bytes:
    """
    Convert binary mask to bytes for download.
    
    Args:
        mask: Binary mask (H, W)
        format: Output format
    
    Returns:
        Mask bytes
    """
    # Ensure it's properly scaled
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    pil_image = Image.fromarray(mask, mode='L')
    
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)
    
    return buffer.getvalue()


def add_legend(
    image: np.ndarray,
    items: list,
    position: str = 'bottom-right'
) -> np.ndarray:
    """
    Add a legend to an image.
    
    Args:
        image: Input image
        items: List of (label, color) tuples
        position: Legend position
    
    Returns:
        Image with legend
    """
    if not items:
        return image
    
    image = image.copy()
    h, w = image.shape[:2]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    # Calculate legend size
    padding = 10
    line_height = 20
    legend_h = len(items) * line_height + 2 * padding
    
    max_text_w = 0
    for label, _ in items:
        (text_w, _), _ = cv2.getTextSize(label, font, font_scale, thickness)
        max_text_w = max(max_text_w, text_w)
    
    legend_w = max_text_w + 40 + 2 * padding
    
    # Position
    if 'right' in position:
        x = w - legend_w - 10
    else:
        x = 10
    
    if 'bottom' in position:
        y = h - legend_h - 10
    else:
        y = 10
    
    # Draw background
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + legend_w, y + legend_h), (255, 255, 255), -1)
    cv2.rectangle(overlay, (x, y), (x + legend_w, y + legend_h), (0, 0, 0), 1)
    
    # Blend
    cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
    
    # Draw items
    for i, (label, color) in enumerate(items):
        item_y = y + padding + i * line_height + 15
        
        # Color box
        cv2.rectangle(
            image,
            (x + padding, item_y - 12),
            (x + padding + 20, item_y + 2),
            color,
            -1
        )
        cv2.rectangle(
            image,
            (x + padding, item_y - 12),
            (x + padding + 20, item_y + 2),
            (0, 0, 0),
            1
        )
        
        # Label
        cv2.putText(
            image,
            label,
            (x + padding + 30, item_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness
        )
    
    return image
