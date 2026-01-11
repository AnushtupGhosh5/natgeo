import torch
import rasterio
import geopandas
import shapely
import fiona
import cv2
import albumentations
import numpy
import matplotlib
import tqdm

print("All libraries imported successfully.")
print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA Available: Yes")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA Available: NO")
