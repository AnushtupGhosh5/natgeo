"""
U-Net architecture with ResNet-34 encoder for semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class DoubleConv(nn.Module):
    """Double convolution block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connection."""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # Handle size mismatch from encoder padding
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    U-Net with ResNet-34 encoder for semantic segmentation.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output channels (default: 1 for binary segmentation)
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        
        # Load ResNet34 backbone (no pretrained weights - we load trained weights)
        base_model = resnet34(weights=None)
        
        if in_channels != 3:
            base_model.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Encoder layers (ResNet stages)
        self.enc1 = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu
        )  # 64 channels, H/2
        
        self.enc2 = nn.Sequential(
            base_model.maxpool,
            base_model.layer1
        )  # 64 channels, H/4
        
        self.enc3 = base_model.layer2  # 128 channels, H/8
        self.enc4 = base_model.layer3  # 256 channels, H/16
        self.bridge = base_model.layer4  # 512 channels, H/32

        # Decoder layers
        self.dec1 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec2 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec3 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.dec4 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)
        
        # Final upsampling and output
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x)         # 64, H/2
        x2 = self.enc2(x1)        # 64, H/4
        x3 = self.enc3(x2)        # 128, H/8
        x4 = self.enc4(x3)        # 256, H/16
        bridge = self.bridge(x4)  # 512, H/32
        
        # Decoder with skip connections
        d1 = self.dec1(bridge, x4)
        d2 = self.dec2(d1, x3)
        d3 = self.dec3(d2, x2)
        d4 = self.dec4(d3, x1)
        
        # Final output
        out = self.final_up(d4)
        out = self.final(out)
        
        return out
