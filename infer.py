import os
import cv2
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import random
import matplotlib.pyplot as plt
import numpy as np
from model import MultiviewFusionModel
from train import HeadpointCoordDataset
from torch.utils.data import DataLoader, random_split


def coords_to_density_map(coords, H, W, sigma=6):
    density = np.zeros((H, W), dtype=np.float32)
    for x, y in coords:
        x_pix = int(np.clip(x, 0, 1) * (W - 1))
        y_pix = int(np.clip(y, 0, 1) * (H - 1))
        if 0 <= x_pix < W and 0 <= y_pix < H:
            density[y_pix, x_pix] = 1
    density = cv2.GaussianBlur(density, (15, 15), sigma)
    if density.sum() > 0:
        density = density * (len(coords) / density.sum())
    return density


if __name__ == "__main__" :
    n_samples = 4  # 출력할 sample 수 지정
    H, W = 256, 256

    #데이터 경로 지정 및 확인
    LEFT_DIR = 'C:/Users/USER/Desktop/dataset/screenshot'
    RIGHT_DIR = 'C:/Users/USER/Desktop/dataset/screenshot'
    CSV_DIR = 'C:/Users/USER/Desktop/dataset/headpoint_csv'

    left_paths = sorted(glob.glob(os.path.join(LEFT_DIR, 'left_*.png')))
    right_paths = sorted(glob.glob(os.path.join(RIGHT_DIR, 'right_*.png')))
    csv_paths = sorted(glob.glob(os.path.join(CSV_DIR, 'headpoint_*.csv')))

    # 전체 80% 학습, 20% 테스트
    dataset = HeadpointCoordDataset(left_paths, right_paths, csv_paths, max_people=200, W=256, H=256)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    #학습 준비 및 시작
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiviewFusionModel().to(device)   

    indices = random.sample(range(len(test_dataset)), n_samples)
    plt.figure(figsize=(10, 4 * n_samples))
    for i, idx in enumerate(indices):
        left_img, right_img, gt_coords, num_people = test_dataset[idx]
        left_img = left_img.unsqueeze(0).to(device)
        right_img = right_img.unsqueeze(0).to(device)
        gt_coords = gt_coords.unsqueeze(0).to(device)
        num_people = int(num_people)
        with torch.no_grad():
            pred_coords = model(left_img, right_img)
        pred = pred_coords[0, :num_people].cpu().numpy()
        gt = gt_coords[0, :num_people].cpu().numpy()
        gt_density = coords_to_density_map(gt, H, W, sigma=6)
        pred_density = coords_to_density_map(pred, H, W, sigma=6)
        
        plt.subplot(n_samples, 2, 2*i+1)
        plt.imshow(gt_density, cmap='jet', origin='lower')
        plt.title(f"[{idx}] GT Density, People: {num_people}")
        plt.axis('off')
        plt.subplot(n_samples, 2, 2*i+2)
        plt.imshow(pred_density, cmap='jet', origin='lower')
        plt.title(f"[{idx}] Pred Density, People: {num_people}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()