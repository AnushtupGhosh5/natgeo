"""
Geo-AI Feature Extraction from Drone Orthophotos
=================================================
A production-ready Streamlit application for extracting geographic features
from drone/satellite orthophotos using deep learning segmentation models.

Supports: Buildings, Roads, Water Bodies, Utilities, Railways

Author: Geo-AI Team
Version: 1.0.0
"""

import streamlit as st
import numpy as np
import rasterio
import torch
import tempfile
import os
from pathlib import Path
from datetime import datetime
import gc

# Import local modules
from config import config, FEATURES, get_available_features, get_feature_options
from utils import (
    InferenceConfig,
    PostProcessConfig,
    load_orthophoto_for_preview,
    run_inference_pipeline,
    run_streaming_inference,
    load_model,
    post_process_mask,
    create_colored_overlay,
    create_mask_visualization,
    normalize_for_display,
    ensure_hwc_format,
    resize_for_display,
    image_to_bytes,
    mask_to_bytes,
    get_device
)


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title=config.page_title,
    page_icon=config.page_icon,
    layout=config.layout,
    initial_sidebar_state="expanded"
)


# ============================================================================
# Custom Styling
# ============================================================================

def inject_custom_css():
    """Inject custom CSS for premium styling."""
    st.markdown("""
    <style>
    /* Main theme */
    :root {
        --primary-color: #1E40AF;
        --secondary-color: #3B82F6;
        --accent-color: #10B981;
        --bg-dark: #0F172A;
        --bg-card: #1E293B;
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1E40AF 0%, #7C3AED 50%, #10B981 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(30, 64, 175, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .info-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    .info-card h3 {
        color: #3B82F6;
        margin-top: 0;
    }
    
    .info-card p {
        color: #94A3B8;
    }
    
    /* Feature card */
    .feature-card {
        background: linear-gradient(135deg, rgba(30, 64, 175, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(30, 64, 175, 0.2);
    }
    
    /* Success/Info boxes */
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(30, 64, 175, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 64, 175, 0.5);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #1E40AF 0%, #10B981 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #3B82F6;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Image containers */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748B;
        border-top: 1px solid rgba(59, 130, 246, 0.1);
        margin-top: 3rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(30, 64, 175, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Session State Management
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'uploaded_file': None,
        'image_metadata': None,
        'preview_image': None,
        'selected_feature': None,
        'model': None,
        'mask': None,
        'overlay': None,
        'inference_complete': False,
        'processing': False,
        'temp_file_path': None,
        'file_size_mb': 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# UI Components
# ============================================================================

def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🌍 Geo-AI Feature Extraction</h1>
        <p>Extract geographic features from drone orthophotos using deep learning</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with controls."""
    with st.sidebar:
        st.markdown("## 📁 Load & Configure")
        st.markdown("---")
        
        # File input - TWO OPTIONS
        st.markdown("### 1️⃣ Select Orthophoto")
        
        # Option 1: Browse existing files in data directory
        # Use absolute path to ensure it works in Streamlit context
        PROJECT_ROOT = Path("/home/anushtup-ghosh/Documents/Projects/natgeo")
        data_dir = PROJECT_ROOT / "data" / "images"
        
        available_files = []
        if data_dir.exists():
            available_files = sorted(list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff")))
        
        input_method = st.radio(
            "Input method",
            ["📂 Browse data folder", "📝 Enter file path"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if input_method == "📂 Browse data folder":
            if available_files:
                # Show files with sizes
                file_options = []
                for f in sorted(available_files):
                    size_gb = f.stat().st_size / (1024**3)
                    file_options.append(f"{f.name} ({size_gb:.2f} GB)")
                
                selected_idx = st.selectbox(
                    "Select orthophoto",
                    range(len(file_options)),
                    format_func=lambda x: file_options[x],
                    help="Select a GeoTIFF from the data/images folder"
                )
                
                if st.button("📂 Load Selected File", use_container_width=True):
                    selected_file = available_files[selected_idx]
                    handle_file_path(str(selected_file))
            else:
                st.warning(f"No .tif/.tiff files found in:\n`{data_dir}`")
        
        else:  # Enter file path
            file_path = st.text_input(
                "Enter full path to GeoTIFF",
                placeholder="/path/to/orthophoto.tif",
                help="Enter the full path to your orthophoto file"
            )
            
            if st.button("📂 Load File", use_container_width=True):
                if file_path and os.path.exists(file_path):
                    handle_file_path(file_path)
                else:
                    st.error("❌ File not found. Please check the path.")
        
        # Show loaded file info
        if st.session_state.temp_file_path:
            st.markdown("---")
            st.success(f"✅ **Loaded:** {Path(st.session_state.temp_file_path).name}")
            if st.session_state.file_size_mb > 0:
                st.caption(f"📦 Size: {st.session_state.file_size_mb:.1f} MB")
        
        st.markdown("---")
        
        # Feature selection
        st.markdown("### 2️⃣ Select Feature Type")
        available_features = get_available_features()
        
        if not available_features:
            st.error("No models found! Please check the models directory.")
            return None
        
        feature_options = list(available_features.keys())
        feature_labels = [f"{FEATURES[f].icon} {FEATURES[f].display_name}" for f in feature_options]
        
        selected_idx = st.selectbox(
            "Feature to extract",
            range(len(feature_options)),
            format_func=lambda x: feature_labels[x],
            help="Select the type of geographic feature to extract"
        )
        
        selected_feature = feature_options[selected_idx]
        st.session_state.selected_feature = selected_feature
        
        # Show feature info
        feat_config = FEATURES[selected_feature]
        st.markdown(f"""
        <div class="feature-card">
            <strong>{feat_config.icon} {feat_config.display_name}</strong><br>
            <small style="color: #94A3B8;">{feat_config.description}</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            st.markdown("**Inference Settings**")
            tile_size = st.select_slider(
                "Tile Size",
                options=[256, 512, 1024],
                value=512,
                help="Size of tiles for processing (512 recommended)"
            )
            batch_size = st.slider(
                "Batch Size",
                min_value=1,
                max_value=16,
                value=4,  # Lower default for memory safety
                help="Number of tiles to process in parallel (lower = less memory)"
            )
            threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05,
                help="Minimum confidence for detection"
            )
            
            st.markdown("**Post-processing**")
            apply_morphology = st.checkbox("Apply morphology", value=True)
            fill_holes = st.checkbox("Fill holes", value=True)
            min_area = st.number_input(
                "Min object area (px)",
                min_value=0,
                max_value=1000,
                value=100
            )
            
            # Store settings
            st.session_state.inference_settings = {
                'tile_size': tile_size,
                'batch_size': batch_size,
                'threshold': threshold,
            }
            st.session_state.postprocess_settings = {
                'apply_morphology': apply_morphology,
                'fill_holes': fill_holes,
                'min_area': min_area,
            }
        
        st.markdown("---")
        
        # Run button
        st.markdown("### 3️⃣ Run Extraction")
        
        can_run = (
            st.session_state.temp_file_path is not None and 
            selected_feature is not None and
            not st.session_state.processing
        )
        
        # Use session state to trigger inference
        if st.button(
            "🚀 Run Model",
            disabled=not can_run,
            use_container_width=True,
            type="primary"
        ):
            st.session_state.run_inference_trigger = True
            st.session_state.inference_feature = selected_feature
            st.rerun()
        
        # Device info
        st.markdown("---")
        device = get_device()
        device_icon = "🔥" if device.type == 'cuda' else "💻"
        st.markdown(f"""
        <div class="info-card">
            <strong>{device_icon} Compute Device</strong><br>
            <span style="color: #10B981;">{device.type.upper()}</span>
            {f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""}
        </div>
        """, unsafe_allow_html=True)
        
        # Memory info
        st.caption("💡 Files are processed in 512×512 tiles to prevent memory issues")
        
        return selected_feature


def handle_file_path(file_path: str):
    """Handle loading file from path (memory-efficient - no upload through browser)."""
    try:
        # Verify file exists and is readable
        if not os.path.exists(file_path):
            st.error(f"❌ File not found: {file_path}")
            return
        
        # Get file size from disk (no loading into memory)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        st.session_state.file_size_mb = file_size_mb
        
        # Store the path directly - NO copying to temp file for large files
        st.session_state.temp_file_path = file_path
        st.session_state.inference_complete = False
        st.session_state.mask = None
        st.session_state.overlay = None
        st.session_state.uploaded_file = None  # Clear old upload state
        
        # Load ONLY a small preview (downsampled) - this is memory safe
        preview, metadata = load_orthophoto_for_preview(
            file_path,
            max_size=config.max_preview_size
        )
        st.session_state.preview_image = preview
        st.session_state.image_metadata = metadata
        
        st.success(f"✅ Loaded: {Path(file_path).name} ({file_size_mb:.1f} MB)")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        import traceback
        st.error(traceback.format_exc())


def render_image_info():
    """Render image information panel."""
    if st.session_state.image_metadata is not None:
        meta = st.session_state.image_metadata
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("Width", f"{meta['width']:,} px")
        with cols[1]:
            st.metric("Height", f"{meta['height']:,} px")
        with cols[2]:
            st.metric("Bands", meta['count'])
        with cols[3]:
            st.metric("CRS", meta['crs'][:20] + "..." if meta['crs'] and len(meta['crs']) > 20 else meta['crs'])


def render_main_panel():
    """Render the main content panel."""
    if st.session_state.preview_image is None:
        # Welcome screen
        st.markdown("""
        <div class="info-card" style="text-align: center; padding: 3rem;">
            <h2>👋 Welcome to Geo-AI Feature Extraction</h2>
            <p style="font-size: 1.1rem; margin: 1.5rem 0;">
                Extract building footprints, roads, water bodies, utilities, and railways 
                from drone orthophotos using state-of-the-art deep learning models.
            </p>
            <p style="color: #3B82F6;">
                <strong>Get started →</strong> Select a GeoTIFF from the sidebar (supports files of ANY size)
            </p>
            <p style="color: #94A3B8; font-size: 0.9rem; margin-top: 1rem;">
                💡 Files are processed in 512×512 tiles - no memory limits!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature showcase
        st.markdown("### 🎯 Available Features")
        cols = st.columns(5)
        
        available = get_available_features()
        for i, (key, feat) in enumerate(FEATURES.items()):
            with cols[i % 5]:
                status = "✅" if key in available else "❌"
                st.markdown(f"""
                <div class="feature-card" style="text-align: center;">
                    <div style="font-size: 2rem;">{feat.icon}</div>
                    <strong>{feat.display_name}</strong><br>
                    <small>{status} {"Available" if key in available else "Model missing"}</small>
                </div>
                """, unsafe_allow_html=True)
        
    else:
        # Image loaded - show preview or results
        render_image_info()
        
        st.markdown("---")
        
        if st.session_state.inference_complete and st.session_state.mask is not None:
            render_results()
        else:
            render_preview()


def render_preview():
    """Render image preview before inference."""
    st.markdown("### 📸 Image Preview")
    
    preview = st.session_state.preview_image
    if preview is not None:
        # Normalize for display
        preview_display = normalize_for_display(preview)
        st.image(
            preview_display,
            caption="Uploaded Orthophoto (Preview)",
            use_container_width=True
        )


def render_results():
    """Render inference results."""
    st.markdown("### 📊 Extraction Results")
    
    feature = st.session_state.selected_feature
    feat_config = FEATURES[feature]
    
    # Results tabs
    tab1, tab2, tab3 = st.tabs(["📷 Side-by-Side", "🎭 Mask", "🔍 Overlay"])
    
    with tab1:
        cols = st.columns(3)
        
        with cols[0]:
            st.markdown("**Original Image**")
            if st.session_state.preview_image is not None:
                st.image(
                    normalize_for_display(st.session_state.preview_image),
                    use_container_width=True
                )
        
        with cols[1]:
            st.markdown(f"**{feat_config.icon} {feat_config.display_name} Mask**")
            mask_viz = create_mask_visualization(
                resize_for_display_mask(st.session_state.mask, st.session_state.preview_image.shape[:2]),
                feature
            )
            st.image(mask_viz, use_container_width=True)
        
        with cols[2]:
            st.markdown("**Overlay**")
            st.image(st.session_state.overlay_display, use_container_width=True)
    
    with tab2:
        st.markdown(f"**{feat_config.display_name} Segmentation Mask**")
        mask_viz = create_mask_visualization(
            resize_for_display_mask(st.session_state.mask, st.session_state.preview_image.shape[:2]),
            feature
        )
        st.image(mask_viz, use_container_width=True)
        
        # Stats
        total_pixels = st.session_state.mask.size
        feature_pixels = np.sum(st.session_state.mask > 0)
        coverage = (feature_pixels / total_pixels) * 100
        
        cols = st.columns(3)
        with cols[0]:
            st.metric("Total Pixels", f"{total_pixels:,}")
        with cols[1]:
            st.metric(f"{feat_config.display_name} Pixels", f"{feature_pixels:,}")
        with cols[2]:
            st.metric("Coverage", f"{coverage:.2f}%")
    
    with tab3:
        st.markdown("**Overlay on Original Image**")
        st.image(st.session_state.overlay_display, use_container_width=True)
    
    # Download section
    st.markdown("---")
    st.markdown("### 💾 Download Results")
    
    # Note about file sizes
    if st.session_state.file_size_mb > 100:
        st.info("💡 For large orthophotos, overlay is shown at preview resolution. Full-resolution mask is available below.")
    
    cols = st.columns(4)
    
    with cols[0]:
        mask_bytes = mask_to_bytes(st.session_state.mask)
        st.download_button(
            label="⬇️ Download Mask (PNG)",
            data=mask_bytes,
            file_name=f"{feature}_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            use_container_width=True
        )
    
    with cols[1]:
        # Use preview overlay (memory efficient)
        overlay_bytes = image_to_bytes(st.session_state.overlay_display)
        st.download_button(
            label="⬇️ Download Overlay (PNG)",
            data=overlay_bytes,
            file_name=f"{feature}_overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            use_container_width=True
        )
    
    with cols[2]:
        mask_viz_full = create_mask_visualization(st.session_state.mask, feature)
        mask_viz_bytes = image_to_bytes(mask_viz_full)
        st.download_button(
            label="⬇️ Download Colored Mask",
            data=mask_viz_bytes,
            file_name=f"{feature}_colored_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            use_container_width=True
        )
    
    with cols[3]:
        st.button(
            "🔄 Run Again",
            on_click=reset_results,
            use_container_width=True
        )


def resize_for_display_mask(mask, target_shape):
    """Resize mask to match display preview size."""
    import cv2
    if mask.shape[:2] != target_shape[:2]:
        return cv2.resize(mask.astype(np.float32), (target_shape[1], target_shape[0])) > 0.5
    return mask


def reset_results():
    """Reset inference results."""
    st.session_state.inference_complete = False
    st.session_state.mask = None
    st.session_state.overlay = None


# ============================================================================
# Inference Pipeline
# ============================================================================

def run_inference(feature_key: str):
    """Run the inference pipeline using streaming for large files."""
    st.session_state.processing = True
    st.session_state.inference_complete = False
    
    feat_config = FEATURES[feature_key]
    raster_path = st.session_state.temp_file_path
    
    # Get settings
    inf_settings = st.session_state.get('inference_settings', {})
    post_settings = st.session_state.get('postprocess_settings', {})
    
    # Create config objects
    inference_config = InferenceConfig(
        tile_size=inf_settings.get('tile_size', 512),
        batch_size=inf_settings.get('batch_size', 8),
        threshold=inf_settings.get('threshold', 0.5),
    )
    
    postprocess_config = PostProcessConfig(
        apply_morphology=post_settings.get('apply_morphology', True),
        fill_holes=post_settings.get('fill_holes', True),
        min_area=post_settings.get('min_area', 100),
    )
    
    # Create progress container
    progress_container = st.empty()
    status_container = st.empty()
    
    try:
        # Load model
        status_container.info(f"🔄 Loading {feat_config.display_name} model...")
        model = load_model(feat_config.model_path)
        
        # Progress callback
        progress_bar = progress_container.progress(0)
        
        def update_progress(progress, stage):
            progress_bar.progress(progress)
            status_container.info(f"🔄 {stage}")
        
        # Run STREAMING inference (memory-efficient for large files)
        # Tiles are read directly from disk, not loaded into memory
        prob_mask, binary_mask = run_streaming_inference(
            model, raster_path, inference_config, update_progress
        )
        
        # Post-processing
        status_container.info("🔄 Post-processing...")
        processed_mask = post_process_mask(binary_mask, postprocess_config)
        
        # Create overlay for PREVIEW only (memory efficient)
        status_container.info("🔄 Creating overlay...")
        preview = st.session_state.preview_image
        preview_mask = resize_for_display_mask(processed_mask, preview.shape[:2])
        overlay_display = create_colored_overlay(preview, preview_mask, feature_key, alpha=0.4)
        
        # Store results (only mask and preview overlay, not full overlay)
        st.session_state.mask = processed_mask
        st.session_state.overlay_display = overlay_display
        st.session_state.inference_complete = True
        
        # Clean up
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        progress_container.empty()
        status_container.success(f"✅ {feat_config.display_name} extraction complete!")
        
    except Exception as e:
        progress_container.empty()
        status_container.error(f"❌ Error during inference: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
    
    finally:
        st.session_state.processing = False


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point."""
    # Initialize
    inject_custom_css()
    init_session_state()
    
    # Check if inference should be triggered
    if st.session_state.get('run_inference_trigger', False):
        st.session_state.run_inference_trigger = False
        feature_key = st.session_state.get('inference_feature')
        if feature_key:
            run_inference(feature_key)
    
    # Render header
    render_header()
    
    # Render sidebar and get selected feature
    selected_feature = render_sidebar()
    
    # Render main panel
    render_main_panel()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>
            <strong>Geo-AI Feature Extraction</strong> | Version 1.0.0<br>
            Built for SVAMITVA / Cadastral Mapping Workflows<br>
            <small>Powered by PyTorch & Streamlit</small>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
