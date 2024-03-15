import math
import os
from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from scipy.stats import multivariate_normal

"""
--------------------------------------------
Hongyi Zheng (github: https://github.com/natezhenghy)
07/Apr/2021
--------------------------------------------
Kai Zhang (github: https://github.com/cszn)
03/Mar/2019
--------------------------------------------
https://github.com/twhui/SRGAN-pyTorch
https://github.com/xinntao/BasicSR
--------------------------------------------
"""

##############
# path utils #
##############

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp',
    '.BMP', '.tif'
]


def is_img(filename: str):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_img_paths(dataroot: np.str0) -> List[str]:
    paths = None  # return None if dataroot is None
    if dataroot is not None:
        paths = sorted(_get_img_paths_from_root(dataroot))
    return paths


def _get_img_paths_from_root(path: str) -> List[str]:
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images: List[str] = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_img(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def makedirs(paths: Union[str, List[str]]):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


def imread_uint(path: str, n_channels: int = 3) -> np.ndarray:
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise NotImplementedError
    return img


def imsave(img: np.ndarray, img_path: str):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


def uint2single(img: np.ndarray) -> np.ndarray:
    return np.float32(img / 255.)


def uint2tensor3(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    tensor: torch.Tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(
        2, 0, 1).float().div(255.)
    return tensor


def tensor2uint(img: torch.Tensor) -> np.ndarray:
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


def single2tensor3(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


def save_d(d: np.ndarray, path: str = ''):
    def merge_images(image_batch: np.ndarray):
        """
            d: C_out, C_in, d_size, d_size
        """
        h, w = image_batch.shape[-2], image_batch.shape[-1]
        img = np.zeros((int(h * 8 + 7), int(w * 8 + 7)))
        for idx, im in enumerate(image_batch):
            i = idx % 8 * (h + 1)
            j = idx // 8 * (w + 1)

            img[j:j + h, i:i + w] = im
        img = cv2.resize(img,
                         dsize=(256, 256),
                         interpolation=cv2.INTER_NEAREST)
        return img

    d = np.where(d > np.quantile(d, 0.75), 0, d)
    d = np.where(d < np.quantile(d, 0.25), 0, d)

    im_merged = merge_images(d)
    im_merged = np.absolute(im_merged)
    plt.imsave(path,
               im_merged,
               cmap='Greys',
               vmin=im_merged.min(),
               vmax=im_merged.max())

def augment_img(img: np.ndarray, mode: int = 0) -> np.ndarray:
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
    else:
        raise ValueError

def calculate_psnr(img1: np.ndarray, img2: np.ndarray, border: int = 0):
    if not img1.shape == img2.shape:
        img2 = img2[..., :img1.shape[-2], :img1.shape[-1]]
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse: float = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray,
                   border: int = 0) -> float:
    if not img1.shape == img2.shape:
        img2 = img2[..., :img1.shape[-2], :img1.shape[-1]]
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims: List[float] = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    s: float = ssim_map.mean()
    return s


def sharpening(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])

    # 使用 filter2D 函数应用锐化核
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image


def cutimg(data):
    _, _, h, w = data.size()
    cut1 = w // 2
    cut2 = h // 2

    img = data.numpy()

    img1_1 = torch.tensor(img[:, :, :cut2, :cut1])
    img1_2 = torch.tensor(img[:, :, :cut2, cut1:])
    img1_3 = torch.tensor(img[:, :, cut2:, :cut1])
    img1_4 = torch.tensor(img[:, :, cut2:, cut1:])

    img_list = [img1_1, img1_2, img1_3, img1_4]
    return img_list


def cating(img_list):
    y0 = img_list[0]
    y1 = img_list[2]
    y2 = img_list[1]
    y3 = img_list[3]

    y5 = torch.cat([y0, y2], dim=3)
    y6 = torch.cat([y1, y3], dim=3)
    y7 = torch.cat([y5, y6], dim=2)

    return y7

def slide_window(data):
    P, Q = 128, 128

    stride = 8

    _, _, H, W = data.size()
    x, y = 0, 0
    img = data.numpy()
    img_list = []
    y_list = []
    x_list = []
    while y < W:
        flag2 = 0
        while x < H:
            flag1 = 0
            if y + Q >= W:
                y = min(y, W - Q)
                flag2 = 1
            if x + P >= H:
                x = min(x, H - P)
                flag1 = 1
            window = torch.tensor(img[:, :, x:x + P, y:y + Q]).to(data.device)
            print(window.shape, x, y, H, W)
            img_list.append(window)
            x_list.append(x)
            y_list.append(y)
            x += stride
            if flag1==1:
                break
        x = 0
        y += stride
        if flag2==1 and flag1==1:
            break
    return {'image': img_list, 'x': x_list, 'y': y_list}, H, W
    
def Slid_cating(data, x_shifts, y_shifts, H, W, i=2):
    if len(data[0].shape) == 2:
        c, h, w = 1, data[0].shape[0], data[0].shape[1]

    elif len(data[0].shape) == 3:
        c, h, w = data[0].shape
    
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    X, Y = np.meshgrid(x, y)

    mean = np.array([w / 2, h / 2])
    cov = np.array([[i * w, 0], [0, i * h]])
    Confid = multivariate_normal.pdf(np.dstack((X, Y)), mean=mean, cov=cov)
    Confid = torch.tensor(Confid).to(data[0].device)
    image = torch.zeros(c, H, W).to(data[0].device)
    Confid_a = torch.zeros(H, W).to(data[0].device)

    for k in range(len(data)):
        image[:, x_shifts[k]:x_shifts[k] + h, y_shifts[k]:y_shifts[k] + w] += data[k] * Confid.unsqueeze(0)
        Confid_a[x_shifts[k]:x_shifts[k] + h, y_shifts[k]:y_shifts[k] + w] += Confid

    image = image / Confid_a.unsqueeze(0)
    return image



