import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.downsample = downsample
        if downsample or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class MultiviewFusionModel(nn.Module):
    def __init__(self, input_H=256, input_W=256, max_people=200):
        super().__init__()
        self.max_people = max_people

        # Input: 6-channel (concatenated RGB left + RGB right)
        self.backbone = nn.Sequential(
            ResidualBlock(6, 32, downsample=True),     # 256 → 128
            ResidualBlock(32, 64, downsample=True),    # 128 → 64
            ResidualBlock(64, 128, downsample=True),   # 64 → 32
        )

        feat_H = input_H // 8
        feat_W = input_W // 8
        feat_dim = feat_H * feat_W * 128

        # Shared feature encoder output → flattened
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.ReLU()
        )

        # Two heads from the shared features
        self.coord_head = nn.Sequential(
            nn.Linear(512, max_people * 2),
            nn.Sigmoid()  # Normalized coordinate output [0, 1]
        )

        self.conf_head = nn.Sequential(
            nn.Linear(512, max_people),
            nn.Sigmoid()  # Confidence per slot [0, 1]
        )

    def forward(self, left_img, right_img):
        x = torch.cat([left_img, right_img], dim=1)  # [B, 6, H, W]
        x = self.backbone(x)                         # [B, 128, H/8, W/8]
        x = self.shared_fc(x)                        # [B, 512]

        coords = self.coord_head(x)                  # [B, max_people * 2]
        confs = self.conf_head(x)                    # [B, max_people]

        coords = coords.view(-1, self.max_people, 2) # [B, N, 2]
        confs = confs.view(-1, self.max_people, 1)   # [B, N, 1]

        output = torch.cat([coords, confs], dim=2)   # [B, N, 3] = (x, y, confidence)
        
        return output

class HeadpointCoordDataset(Dataset) :
    def __init__(self, left_paths, right_paths, csv_paths, transform=None, max_people=200, W=256, H=256) :
        self.left_paths = left_paths
        self.right_paths = right_paths
        self.csv_paths = csv_paths
        
        self.transform = transform
        self.max_people = max_people
        
        self.W, self.H = W, H
        
        # 전체 x/y min/max 찾기 (모든 CSV 파일에서)
        self.x_min, self.x_max, self.y_min, self.y_max = -9, 9, -1.5, 18.5

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx) :
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
        if self.transform :
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        else :
            to_tensor = T.ToTensor()
            left_img = to_tensor(left_img)
            right_img = to_tensor(right_img)
        
        coords = torch.tensor(coords, dtype=torch.float32)
        
        return left_img, right_img, coords, num_people

if __name__ == "__main__" :
    pass