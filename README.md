# 🌍 Geo-AI Feature Extraction

A production-ready deep learning application for semantic segmentation of geospatial features from drone orthophotos and satellite imagery.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Overview

This project provides a robust, user-friendly interface for extracting geographic features such as buildings, roads, water bodies, and railways from large-scale orthophotos. Built with Streamlit and PyTorch, it utilizes state-of-the-art semantic segmentation models (U-Net with ResNet encoders) to deliver high-accuracy results while handling massive geospatial files efficiently.

## ✨ Key Features

- **🏗 Multi-Class Extraction**: Support for multiple feature types including:
  - Buildings
  - Roads
  - Water Bodies
  - Railways
  - Utilities
- **🚀 Memory-Efficient Processing**: specific "Streaming Inference" pipeline designed to handle large GeoTIFFs (1GB+) without crashing memory, processing images in moving windows.
- **🎨 Premium UI/UX**: A modern, dark-themed interface with custom styling, smooth transitions, and intuitive controls.
- **⚙️ Advanced Configuration**:
  - Adjustable tile sizes (256, 512, 1024)
  - Configurable batch sizes and confidence thresholds
  - Post-processing options (morphology, hole filling)
- **🗺️ Geospatial Precision**: Preserves origin coordinate reference systems (CRS) and geotransforms in all output files.
- **⚡ GPU Acceleration**: Automatic CUDA detection for accelerated inference on supported hardware.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Deep Learning**: PyTorch, Torchvision
- **Geospatial Processing**: Rasterio, GeoPandas, Shapely
- **Image Processing**: OpenCV (cv2), NumPy, Pillow

## 📂 Directory Structure

```
natgeo/
├── app/                 # Main application source code
│   ├── app.py           # Entry point
│   ├── config.py        # Configuration settings
│   ├── utils/           # Utility functions (inference, processing)
│   └── .streamlit/      # Streamlit configuration
├── data/                # Data storage
│   └── images/          # Input orthophotos
├── models/              # Saved model weights (.pth files)
├── notebooks/           # Jupyter notebooks for model training & experiments
├── outputs/             # Inference results
└── requirements.txt     # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for faster processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/geo-ai-extraction.git
   cd geo-ai-extraction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r app/requirements.txt
   ```

### Running the Application

1. **Start the Streamlit server**
   ```bash
   streamlit run app/app.py
   ```

2. **Access the UI**
   Open your browser and navigate to `http://localhost:8501`.

3. **Usage**
   - **Load Data**: Browse for a `.tif` file in your `data/` directory or enter a direct file path.
   - **Select Feature**: Choose the target feature (e.g., Water Body, Building) from the sidebar.
   - **Configure**: Adjust inference settings if needed (Tile Size, Confidence).
   - **Run**: Click "Run Model" to start the extraction process.
   - **Export**: Download the binary masks, overlays, or colored visualization maps.

## 📊 Model Details

The application uses U-Net architectures with pre-trained encoders (e.g., ResNet-34) trained on specific datasets for each feature type.
- **Input**: 3-channel RGB Orthophotos
- **Output**: Binary segmentation masks

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
