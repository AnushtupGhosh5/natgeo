# 🌍 Geo-AI Feature Extraction: Deep Learning Pipelines

This repository houses a collection of high-performance **Research & Development notebooks** designed to extract critical geospatial features from high-resolution drone orthophotos.

Using state-of-the-art Computer Vision techniques, we process massive geospatial datasets (GeoTIFFs) to identify and segment infrastructure and natural features.

## 🧠 Core Research Notebooks

The primary value of this project lies in the specialized Jupyter notebooks, each tailored for a specific feature extraction challenge. These notebooks handle the full pipeline: **Data Ingestion** -> **Preprocessing (Tiling/Rasterization)** -> **Model Training** -> **Inference**.

| Notebook | Feature Type | Description |
|----------|-------------|-------------|
| 🏢 **`building.ipynb`** | **Building Footprints** | End-to-end pipeline for segmenting building rooftops. Features strict mask alignment and U-Net training with ResNet encoders. |
| 🛣️ **`main.ipynb`** | **Road Network** | Rasterizes linear road vectors into masks and trains a model to detect road surfaces/centerlines, handling complex connectivity. |
| 💧 **`waterbodies.ipynb`** | **Water Bodies** | Semantic segmentation for ponds, potential lakes, and water storage. Optimized to reduce false positives in dry terrain. |
| 🚂 **`railways.ipynb`** | **Railway Tracks** | Specialized pipeline for linear feature extraction, preserving the connectivity of railway segments over long distances. |
| ⚡ **`utilities.ipynb`** | **Utilities** | Detection of smaller utility infrastructure. Handles high-imbalance classes where features are small relative to the image size. |

## 🛠️ Technical Methodology

Our pipelines address the unique challenges of geospatial deep learning:

### 1. Large-Scale Data Handling
- **Tiling**: Images are too large for GPU memory (1GB+). We slice them into fixed-size chips (e.g., 512x512) with overlap to prevent edge artifacts.
- **Geospatial Alignment**: All vector data (Shapefiles) are strictly reprojected to match the Orthophoto's Coordinate Reference System (CRS) before rasterization.

### 2. Deep Learning Architecture
- **Models**: We primarily use **U-Net** architectures with **ResNet-34** or **ResNet-50** encoders pretrained on ImageNet.
- **Loss Functions**: Custom implementations of **Dice Loss + Binary Cross Entropy (BCE)** to handle class imbalance (e.g., roads are only ~5% of the pixels).
- **Inference**: GPU-accelerated variable-window inference.

## 🚀 Getting Started

The notebooks are the best way to understand and run the code.

### Prerequisites
- Python 3.9+
- NVIDIA GPU (Recommended for training)
- 16GB+ RAM (For handling large GeoTIFFs)

### Setup

1. **Install Dependencies**
   ```bash
   pip install -r app/requirements.txt
   ```

2. **Launch Jupyter**
   ```bash
   jupyter lab
   ```

3. **Data Structure**
   Ensure your data is organized as follows for the notebooks to work out-of-the-box:
   ```
   natgeo/
   ├── data/
   │   ├── images/          # Place huge .tif orthophotos here
   │   └── shp/             # Corresponding .shp shapefiles
   ```

## 📱 Streamlit Visualization (Demo)

*Note: The Streamlit app acts as a visualization layer and quick demo.*

While the notebooks contain the rigorous training cycles and full pipelines, an interactive web app is included for strictly **inference and visualization** purposes.

To run the visualization demo:
```bash
streamlit run app/app.py
```

## 🤝 Contributing

This is an active research project. If you improve a pipeline (e.g., better loss function for thin features like railways), please submit a Pull Request.
