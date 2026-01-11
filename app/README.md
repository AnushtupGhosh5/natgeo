# Geo-AI Feature Extraction from Drone Orthophotos

A production-ready Streamlit web application for extracting geographic features from drone/satellite orthophotos using deep learning segmentation models.

## 🎯 Features

- **Buildings** - Extract building footprints
- **Roads** - Extract road networks
- **Water Bodies** - Extract ponds, lakes, and rivers
- **Utilities** - Extract vegetation, overhead tanks, wells, transformers
- **Railways** - Extract railway lines

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for faster inference)
- Trained model weights in the `models/` directory

### Installation

1. Navigate to the app directory:
```bash
cd app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
app/
├── app.py              # Main Streamlit application
├── config.py           # Configuration and feature definitions
├── requirements.txt    # Python dependencies
├── models/
│   ├── __init__.py
│   └── unet.py         # U-Net architecture with ResNet-34 encoder
└── utils/
    ├── __init__.py
    ├── tiling.py       # Image tiling and stitching utilities
    ├── inference.py    # Model inference engine
    ├── postprocessing.py # Mask post-processing utilities
    └── visualization.py # Display and export utilities
```

## 🔧 How It Works

### Workflow

1. **Upload** - Upload a GeoTIFF orthophoto (.tif/.tiff)
2. **Select** - Choose the feature type to extract
3. **Configure** - Adjust inference settings (optional)
4. **Run** - Click "Run Model" to start extraction
5. **Review** - View results in side-by-side comparison
6. **Download** - Export mask and overlay images

### Technical Pipeline

1. **Load Image** - Read GeoTIFF using rasterio, preserving CRS
2. **Tile** - Split orthophoto into 512×512 tiles
3. **Normalize** - Scale pixel values to [0, 1]
4. **Inference** - Run batched inference (batch_size=8)
5. **Stitch** - Reassemble tiles into full-resolution mask
6. **Post-process** - Apply morphology, fill holes, filter by area
7. **Visualize** - Generate colored overlay

### GPU/CPU Fallback

The application automatically detects CUDA availability:
- **GPU Available**: Uses CUDA for faster inference
- **CPU Only**: Falls back to CPU inference (slower but works)

## ⚙️ Configuration

### Inference Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Tile Size | 512 | Size of processing tiles |
| Batch Size | 8 | Tiles per batch (GPU memory) |
| Threshold | 0.5 | Confidence threshold |

### Post-processing Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Apply Morphology | True | Clean up mask edges |
| Fill Holes | True | Fill interior holes |
| Min Area | 100 | Remove small detections |

## 📊 Model Requirements

Models should be PyTorch state dictionaries for the U-Net architecture:
- **Input**: 3-channel RGB (512×512 tiles)
- **Output**: 1-channel binary mask
- **Encoder**: ResNet-34 (ImageNet pre-trained optional)

### Expected Model Files

```
models/
├── building_unet.pth
├── road_unet.pth (or in notebooks/models/)
├── best_water_model.pth
├── utility_unet.pth
└── railway_unet.pth
```

## 🏛️ Use Case

This application is designed for:
- **SVAMITVA** scheme implementations
- **Cadastral mapping** workflows
- **Rural/urban planning** applications
- **Disaster response** mapping
- **Agricultural monitoring**

## 🔒 Notes

- Large orthophotos are processed tile-by-tile to manage memory
- Progress indicators show real-time inference status
- Results can be downloaded as PNG for GIS integration

## 📄 License

Internal use only - Government of India projects

---

**Built with PyTorch & Streamlit** | Version 1.0.0
