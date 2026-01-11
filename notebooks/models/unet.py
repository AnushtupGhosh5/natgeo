import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Matches 'up.weight'
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # Matches 'conv.conv1...'
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        
        # Determine if we need to resize due to padding issues in encoder
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Load ResNet34
        # We use weights=None to match the architecture (loaded state dict is random or trained)
        # Note: 'weights' arg is newer torchvision, use 'pretrained=False' or simply construct.
        # But 'weights=None' is safe in new versions too.
        # Actually user has 0.25.0 dev, so weights=None or ResNet34_Weights.DEFAULT
        # The user loads weights manually, so initialization doesn't matter much.
        base_model = resnet34(weights=None)
        
        if in_channels != 3:
            base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Encoder Layers matching keys
        
        # enc1: Stem
        # keys: enc1.0 (conv1), enc1.1 (bn1)
        # No relu key (it's stateless), but it's part of the sequence logic.
        self.enc1 = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu
        )
        
        # enc2: MaxPool + Layer1
        # keys: enc2.1 (layer1). So index 0 must be maxpool.
        self.enc2 = nn.Sequential(
            base_model.maxpool,
            base_model.layer1
        )
        
        # enc3 -> layer2
        self.enc3 = base_model.layer2
        
        # enc4 -> layer3
        self.enc4 = base_model.layer3
        
        # bridge -> layer4
        self.bridge = base_model.layer4

        # Decoder Layers
        # Channels:
        # enc1: 64
        # enc2: 64
        # enc3: 128
        # enc4: 256
        # bridge: 512
        
        self.dec1 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec2 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec3 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.dec4 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)
        
        # Final upsampling
        # Matches 'final_up.weight'
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        # Final conv
        # Matches 'final.weight'
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Enum encoder outputs
        x1 = self.enc1(x)         # 64, H/2
        x2 = self.enc2(x1)        # 64, H/4
        x3 = self.enc3(x2)        # 128, H/8
        x4 = self.enc4(x3)        # 256, H/16
        bridge = self.bridge(x4)  # 512, H/32
        
        # Decoder
        d1 = self.dec1(bridge, x4)
        d2 = self.dec2(d1, x3)
        d3 = self.dec3(d2, x2)
        d4 = self.dec4(d3, x1)
        
        # Final
        out = self.final_up(d4)
        out = self.final(out)
        
        return out
