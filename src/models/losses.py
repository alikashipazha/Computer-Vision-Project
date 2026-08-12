import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelEdgeLoss(nn.Module):
    """
    Calculates L1 Loss on Sobel edge maps to focus on text boundaries and legibility.
    """
    def __init__(self):
        super(SobelEdgeLoss, self).__init__()
        # Define Sobel kernels
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert RGB images to Grayscale using standard weights
        pred_gray = 0.299 * pred[:, 0:1, :, :] + 0.587 * pred[:, 1:2, :, :] + 0.114 * pred[:, 2:3, :, :]
        target_gray = 0.299 * target[:, 0:1, :, :] + 0.587 * target[:, 1:2, :, :] + 0.114 * target[:, 2:3, :, :]
        
        # Defensive check to ensure filters match input device dynamically
        device_kernel_x = self.kernel_x.to(pred_gray.device)
        device_kernel_y = self.kernel_y.to(pred_gray.device)
        
        # Apply Sobel filters
        pred_dx = F.conv2d(pred_gray, device_kernel_x, padding=1)
        pred_dy = F.conv2d(pred_gray, device_kernel_y, padding=1)
        
        target_dx = F.conv2d(target_gray, device_kernel_x, padding=1)
        target_dy = F.conv2d(target_gray, device_kernel_y, padding=1)
        
        # Calculate edge magnitudes
        pred_edge = torch.sqrt(pred_dx**2 + pred_dy**2 + 1e-8)
        target_edge = torch.sqrt(target_dx**2 + target_dy**2 + 1e-8)
        
        return F.l1_loss(pred_edge, target_edge)


class SSIMLoss(nn.Module):
    """
    Pure PyTorch implementation of Structural Similarity Index (SSIM) Loss.
    Designed from scratch to avoid external dependency issues.
    """
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        
        # Generate 1D Gaussian window
        gaussian = torch.tensor([
            torch.exp(-torch.tensor((x - window_size // 2) ** 2) / float(2 * 1.5 ** 2))
            for x in range(window_size)
        ], dtype=torch.float32)
        gaussian = gaussian / gaussian.sum()
        
        # Create 2D Gaussian window
        _1D_window = gaussian.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(self.channel, 1, window_size, window_size).contiguous()
        self.register_buffer('window', window)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # Defensive check to ensure window matches input device dynamically
        device_window = self.window.to(img1.device)
        
        mu1 = F.conv2d(img1, device_window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, device_window, padding=self.window_size//2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, device_window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, device_window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, device_window, padding=self.window_size//2, groups=self.channel) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        if self.size_average:
            return 1.0 - ssim_map.mean()
        else:
            return 1.0 - ssim_map.mean(1).mean(1).mean(1)


class CompositeLoss(nn.Module):
    """
    Combined Loss Function: Alpha * L1_Loss + Beta * SSIM_Loss + Gamma * Sobel_Edge_Loss
    Balances structural integrity, pixel-level color, and edge contrast.
    """
    def __init__(self, alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.2):
        super(CompositeLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.edge = SobelEdgeLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_l1 = self.l1(pred, target)
        loss_ssim = self.ssim(pred, target)
        loss_edge = self.edge(pred, target)
        
        return self.alpha * loss_l1 + self.beta * loss_ssim + self.gamma * loss_edge