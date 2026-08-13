import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm2d -> ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DirectRegressionNet(nn.Module):
    """
    Approach A: Direct coordinate regression.
    Regularized with 1D Feature Dropout inside the fully connected classifier head.
    """
    def __init__(self, in_channels: int = 3):
        super(DirectRegressionNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 256
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # 16
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # Phase 6: Standard dropout on flattened features
            nn.Linear(256, 8),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.fc(features)

class HeatmapUNet(nn.Module):
    """
    Approach B: Corner heatmap regression.
    Regularized with 2D Spatial Dropout at the deepest bottleneck layer.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 4):
        super(HeatmapUNet, self).__init__()
        
        # Encoder (Downsampling path)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # Phase 6: Spatial Dropout at the bottleneck
        self.dropout = nn.Dropout2d(p=0.5)
        
        # Decoder (Upsampling path with Skip Connections)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Apply Spatial Dropout at bottleneck
        x4 = self.dropout(x4)
        
        # Decoder
        x_up = self.up1(x4)
        x_concat = torch.cat([x_up, x3], dim=1)
        x_dec1 = self.conv_up1(x_concat)
        
        x_up = self.up2(x_dec1)
        x_concat = torch.cat([x_up, x2], dim=1)
        x_dec2 = self.conv_up2(x_concat)
        
        x_up = self.up3(x_dec2)
        x_concat = torch.cat([x_up, x1], dim=1)
        x_dec3 = self.conv_up3(x_concat)
        
        logits = self.outc(x_dec3)
        return self.sigmoid(logits)
