import os
import glob
import torch
import transformers
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader, random_split
from model import MultiviewFusionModel

class HeadpointCoordDataset(Dataset):
    def __init__(self, left_paths, right_paths, csv_paths, transform=None, max_people=200, W=256, H=256):
        self.left_paths = left_paths
        self.right_paths = right_paths
        self.csv_paths = csv_paths
        self.transform = transform
        self.max_people = max_people
        self.W, self.H = W, H
        # x,y min/max 값 지정
        self.x_min, self.x_max, self.y_min, self.y_max = -9, 9, -1.5, 18.5

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_img = Image.open(self.left_paths[idx]).convert('RGB').resize((self.W, self.H))
        right_img = Image.open(self.right_paths[idx]).convert('RGB').resize((self.W, self.H))
        df = pd.read_csv(self.csv_paths[idx])
        # 좌표 정규화(0~1)
        xs = (df['x'].values - self.x_min) / (self.x_max - self.x_min)
        ys = (df['y'].values - self.y_min) / (self.y_max - self.y_min)
        coords = np.stack([xs, ys], axis=1)
        num_people = coords.shape[0]
        # max_people 기준 패딩
        if num_people < self.max_people:
            padding = np.zeros((self.max_people - num_people, 2))
            coords = np.vstack([coords, padding])
        else:
            coords = coords[:self.max_people]
        # transform (ToTensor)
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        else:
            import torchvision.transforms as T
            to_tensor = T.ToTensor()
            left_img = to_tensor(left_img)
            right_img = to_tensor(right_img)
        coords = torch.tensor(coords, dtype=torch.float32)
        return left_img, right_img, coords, num_people

def hungarian_loss(pred_coords, gt_coords, num_people):
    """
    pred_coords, gt_coords: (B, max_people, 2)
    num_people: (B,) 실제 사람 수
    """
    loss = 0.0
    batch_size = pred_coords.shape[0]
    device = pred_coords.device
    dtype = pred_coords.dtype
    for i in range(batch_size):
        n = int(num_people[i])
        if n == 0:
            continue
        pred = pred_coords[i, :n]  # (n, 2)
        gt = gt_coords[i, :n]      # (n, 2)
        cost = torch.cdist(pred, gt, p=2)  # (n, n)  (torch tensor)
        row_ind, col_ind = linear_sum_assignment(cost.cpu().detach().numpy())
        pairwise_loss = cost[row_ind, col_ind].sum() / n  # ← torch tensor!
        loss += pairwise_loss
        
    return loss / batch_size


if __name__ == "__main__" :

    #데이터 경로 지정 및 확인
    LEFT_DIR = 'C:/Users/USER/Desktop/dataset/screenshot'
    RIGHT_DIR = 'C:/Users/USER/Desktop/dataset/screenshot'
    CSV_DIR = 'C:/Users/USER/Desktop/dataset/headpoint_csv'

    left_paths = sorted(glob.glob(os.path.join(LEFT_DIR, 'left_*.png')))
    right_paths = sorted(glob.glob(os.path.join(RIGHT_DIR, 'right_*.png')))
    csv_paths = sorted(glob.glob(os.path.join(CSV_DIR, 'headpoint_*.csv')))

    print(f"Left: {len(left_paths)}, Right: {len(right_paths)}, CSV: {len(csv_paths)}")
    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for left_img, right_img, gt_coords, num_people in pbar:
                left_img = left_img.to(device)
                right_img = right_img.to(device)
                gt_coords = gt_coords.to(device)
                num_people = num_people.to(device)
                optimizer.zero_grad()
                pred_coords = model(left_img, right_img)
                loss = hungarian_loss(pred_coords, gt_coords, num_people)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), f'coordreg_checkpoint_epoch_{epoch+1}.pth')