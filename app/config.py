"""
Configuration for the Geo-AI Feature Extraction Application.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional


# Base paths
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class FeatureConfig:
    """Configuration for a single feature type."""
    name: str
    display_name: str
    model_path: str
    color: tuple  # RGB
    description: str
    icon: str  # Emoji


# Feature configurations
FEATURES: Dict[str, FeatureConfig] = {
    "buildings": FeatureConfig(
        name="buildings",
        display_name="Buildings",
        model_path=str(MODELS_DIR / "building_unet.pth"),
        color=(255, 120, 50),
        description="Extract building footprints from orthophotos",
        icon="🏠"
    ),
    "roads": FeatureConfig(
        name="roads",
        display_name="Roads",
        model_path=str(PROJECT_ROOT / "notebooks" / "models" / "road_unet.pth"),
        color=(64, 64, 64),
        description="Extract road networks from orthophotos",
        icon="🛤️"
    ),
    "water_bodies": FeatureConfig(
        name="water_bodies",
        display_name="Water Bodies",
        model_path=str(MODELS_DIR / "best_water_model.pth"),
        color=(30, 144, 255),
        description="Extract water bodies (ponds, lakes, rivers) from orthophotos",
        icon="💧"
    ),
    "utilities": FeatureConfig(
        name="utilities",
        display_name="Utilities",
        model_path=str(MODELS_DIR / "utility_unet.pth"),
        color=(34, 139, 34),
        description="Extract utilities (vegetation, overhead tanks, wells, transformers)",
        icon="⚡"
    ),
    "railways": FeatureConfig(
        name="railways",
        display_name="Railways",
        model_path=str(MODELS_DIR / "railway_unet.pth"),
        color=(139, 69, 19),
        description="Extract railway lines from orthophotos",
        icon="🚂"
    ),
}


@dataclass
class InferenceSettings:
    """Settings for model inference."""
    tile_size: int = 512
    batch_size: int = 8
    overlap: int = 0
    threshold: float = 0.5


@dataclass
class PostProcessSettings:
    """Settings for post-processing."""
    apply_morphology: bool = True
    kernel_size: int = 3
    min_area: int = 100
    fill_holes: bool = True
    smooth_boundaries: bool = True


@dataclass
class AppConfig:
    """Main application configuration."""
    app_name: str = "Geo-AI Feature Extraction"
    app_version: str = "1.0.0"
    page_title: str = "Geo-AI Feature Extraction | SVAMITVA"
    page_icon: str = "🌍"
    layout: str = "wide"
    
    # Display settings
    max_preview_size: int = 1024
    max_display_width: int = 800
    
    # File settings
    allowed_extensions: tuple = (".tif", ".tiff")
    max_file_size_mb: int = 500
    
    # Inference settings
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    
    # Post-processing settings
    postprocess: PostProcessSettings = field(default_factory=PostProcessSettings)


# Default configuration instance
config = AppConfig()


def get_available_features() -> Dict[str, FeatureConfig]:
    """
    Get only features that have available model files.
    
    Returns:
        Dictionary of available feature configurations
    """
    available = {}
    for key, feat in FEATURES.items():
        if os.path.exists(feat.model_path):
            available[key] = feat
    return available


def get_feature_options() -> list:
    """
    Get list of feature options for dropdown.
    
    Returns:
        List of tuples (display_name, key)
    """
    available = get_available_features()
    return [(f"{feat.icon} {feat.display_name}", key) for key, feat in available.items()]
