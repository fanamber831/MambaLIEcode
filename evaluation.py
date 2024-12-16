import glob
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import cv2
import argparse

from natsort import natsort
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

class Measure():
    def __init__(self, net='alex', use_gpu=False):
        self.device = 'cuda' if use_gpu else 'cpu'
        self.model = lpips.LPIPS(net=net)
        self.model.to(self.device)

    def measure(self, imgA, imgB):
        return [self.psnr(imgA, imgB), self.ssim(imgA, imgB), self.lpips(imgA, imgB)]

    def lpips(self, imgA, imgB, model=None):
        tA = t(imgA).to(self.device)
        tB = t(imgB).to(self.device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB):
        # Calculate the minimum size for win_size, ensuring it is odd and less than the smallest dimension of the image
        win_size = min(imgA.shape[0], imgA.shape[1], 7)
        if win_size % 2 == 0:
            win_size -= 1

        score, diff = ssim(imgA, imgB, full=True, multichannel=True, win_size=win_size, channel_axis=2)
        return score

    def psnr(self, imgA, imgB):
        psnr_val = psnr(imgA, imgB)
        return psnr_val

def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1

def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))

def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]

def format_result(psnr, ssim, lpips):
    return f'{psnr:.5f}, {ssim:.5f}, {lpips:.5f}'

def measure_dirs(dirA, dirB, img_type, use_gpu, verbose=False):
    if verbose:
        vprint = lambda x: print(x)
    else:
        vprint = lambda x: None

    t_init = time.time()

    paths_A = fiFindByWildcard(os.path.join(dirA, f'*.{img_type}'))
    paths_B = fiFindByWildcard(os.path.join(dirB, f'*.{img_type}'))

    vprint("Comparing: ")
    vprint(dirA)
    vprint(dirB)

    measure = Measure(use_gpu=use_gpu)

    results = []
    for pathA, pathB in zip(paths_A, paths_B):
        result = OrderedDict()

        t = time.time()
        imgA = imread(pathA)
        imgB = imread(pathB)
        psnr_val, ssim_val, lpips_val = measure.measure(imgA, imgB)
        result['psnr'] = psnr_val
        result['ssim'] = ssim_val
        result['lpips'] = lpips_val
        d = time.time() - t
        vprint(f"{pathA.split('/')[-1]}, {pathB.split('/')[-1]}, {format_result(psnr_val, ssim_val, lpips_val)}, {d:0.1f}")

        results.append(result)

    psnr_avg = np.mean([result['psnr'] for result in results])
    ssim_avg = np.mean([result['ssim'] for result in results])
    lpips_avg = np.mean([result['lpips'] for result in results])

    vprint(f"Final Result: {format_result(psnr_avg, ssim_avg, lpips_avg)}, {time.time() - t_init:0.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dirA', default='./datasets/LOL/test/high/', type=str)
    parser.add_argument('-dirB', default='./results/LOL/', type=str)
    parser.add_argument('-type', default='png')
    parser.add_argument('--use_gpu', default=True)
    args = parser.parse_args()

    dirA = args.dirA
    dirB = args.dirB
    img_type = args.type
    use_gpu = args.use_gpu

    if len(dirA) > 0 and len(dirB) > 0:
        measure_dirs(dirA, dirB, img_type, use_gpu=use_gpu, verbose=True)
