"""Utility modules for Geo-AI feature extraction."""

from .tiling import (
    TileInfo,
    calculate_tile_grid,
    extract_tiles_from_array,
    extract_tiles_from_rasterio,
    stitch_tiles,
    load_orthophoto_for_preview
)
from .inference import (
    InferenceConfig,
    get_device,
    run_batched_inference,
    run_inference_pipeline,
    run_streaming_inference,
    load_model
)
from .postprocessing import (
    PostProcessConfig,
    apply_morphological_ops,
    remove_small_components,
    fill_holes,
    smooth_boundaries,
    post_process_mask,
    get_feature_color,
    create_colored_overlay,
    create_mask_visualization
)
from .visualization import (
    normalize_for_display,
    ensure_hwc_format,
    resize_for_display,
    create_side_by_side,
    image_to_bytes,
    mask_to_bytes
)

__all__ = [
    # Tiling
    'TileInfo',
    'calculate_tile_grid',
    'extract_tiles_from_array',
    'extract_tiles_from_rasterio',
    'stitch_tiles',
    'load_orthophoto_for_preview',
    # Inference
    'InferenceConfig',
    'get_device',
    'run_batched_inference',
    'run_inference_pipeline',
    'run_streaming_inference',
    'load_model',
    # Post-processing
    'PostProcessConfig',
    'apply_morphological_ops',
    'remove_small_components',
    'fill_holes',
    'smooth_boundaries',
    'post_process_mask',
    'get_feature_color',
    'create_colored_overlay',
    'create_mask_visualization',
    # Visualization
    'normalize_for_display',
    'ensure_hwc_format',
    'resize_for_display',
    'create_side_by_side',
    'image_to_bytes',
    'mask_to_bytes'
]
