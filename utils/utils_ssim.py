import torch
import torch.nn.functional as F
import cv2
import numpy as np


def calculate_ssim2(img1: torch.Tensor, img2: torch.Tensor,
                   border: int = 0) -> float:
    if not img1.shape == img2.shape:
        img2 = img2[..., :img1.shape[-2], :img1.shape[-1]]
    h, w = img1.shape[-2:]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]


    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims: torch.Tensor = torch.zeros(3, device=img1.device)
            for i in range(3):
                ssims[i] = ssim(img1[i], img2[i])
            return ssims.mean()
        elif img1.shape[0] == 1:
            return ssim(img1.squeeze(), img2.squeeze())
        else:
            raise ValueError('Wrong input image dimensions.')
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.to(torch.float64)*255
    img2 = img2.to(torch.float64)*255
    kernel = cv2.getGaussianKernel(11, 1.5)
    kernel = torch.from_numpy(np.outer(kernel, kernel.transpose())).to(img1.device, torch.float64)

    mu1 = F.conv2d(img1[None, None], kernel[None, None])[0, 0, 5:-5, 5:-5]
    mu2 = F.conv2d(img2[None, None], kernel[None, None])[0, 0, 5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1[None, None]**2, kernel[None, None])[0, 0, 5:-5, 5:-5] - mu1_sq
    sigma2_sq = F.conv2d(img2[None, None]**2, kernel[None, None])[0, 0, 5:-5, 5:-5] - mu2_sq
    sigma12 = F.conv2d(img1[None, None] * img2[None, None], kernel[None, None])[0, 0, 5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    s: float = ssim_map.mean()
    return s

