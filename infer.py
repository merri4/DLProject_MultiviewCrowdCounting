import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os

import argparse

from model import MultiviewFusionModel, HeadpointCoordDataset


def load_checkpoint(filename, model, optimizer=None):  
    
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])  
    
    if optimizer is not None :
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print("Model {} Loaded! Loss {} on Epoch {}".format(filename, loss, epoch))
    return epoch, loss


def coords_to_density_map(coords, H, W, sigma=6) :
    density = np.zeros((H, W), dtype=np.float32)
    for x, y in coords :
        x_pix = int(np.clip(x, 0, 1) * (W - 1))
        y_pix = int(np.clip(y, 0, 1) * (H - 1))
        if 0 <= x_pix < W and 0 <= y_pix < H:
            density[y_pix, x_pix] = 1
    density = cv2.GaussianBlur(density, (15, 15), sigma)
    if density.sum() > 0:
        density = density * (len(coords) / density.sum())
    return density

def parse_arguments() :

    parser = argparse.ArgumentParser(description="Multiview Crowd Counting")
    
    # environmental settings
    parser.add_argument("--data_path", type=str, default='./dataset/')
    parser.add_argument("--model_path", type=str, default='./output/epoch_1.pth')

    # hyper-parameters
    parser.add_argument("--train_split_rate", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_people", type=int, default=200)

    args = parser.parse_args()

    return args


if __name__ == "__main__" :

    args = parse_arguments()

    # ======================================================
    # 이미지 넣어서 데이터 불러오기
    # ======================================================

    left_paths = sorted(glob.glob(os.path.join(args.data_path, 'screenshot/left_*.png')))
    right_paths = sorted(glob.glob(os.path.join(args.data_path, 'screenshot/right_*.png')))
    csv_paths = sorted(glob.glob(os.path.join(args.data_path, 'headpoint_csv/headpoint_*.csv')))

    print(f"Left: {len(left_paths)}, Right: {len(right_paths)}, CSV: {len(csv_paths)}")

    dataset = HeadpointCoordDataset(left_paths, right_paths, csv_paths, max_people=args.max_people, W=256, H=256)
    
    n_total = len(dataset)
    n_train = int(args.train_split_rate * n_total)
    n_test = n_total - n_train

    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # =========================================================
    # Model Prep
    # =========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiviewFusionModel().to(device)
    load_checkpoint(args.model_path, model)


    # ======================================================
    # 시각화
    # ======================================================
    n_samples = 4  # 출력할 sample 수 지정
    H, W = 256, 256

    indices = random.sample(range(len(test_dataset)), n_samples)
    plt.figure(figsize=(10, 4 * n_samples))

    for i, idx in enumerate(indices) :
        
        left_img, right_img, gt_coords, num_people = test_dataset[idx]
        
        left_img = left_img.unsqueeze(0).to(device)
        right_img = right_img.unsqueeze(0).to(device)
        gt_coords = gt_coords.unsqueeze(0).to(device)
        
        num_people = int(num_people)
        
        with torch.no_grad():
            pred_coords = model(left_img, right_img)
        
        gt = gt_coords[0, :num_people].cpu().numpy()
        gt_density = coords_to_density_map(gt, H, W, sigma=6)
        
        pred = pred_coords[0, :num_people].cpu().numpy()
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
