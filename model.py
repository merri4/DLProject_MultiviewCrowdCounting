import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

class MultiviewFusionModel(nn.Module) :
    def __init__(self, input_H=256, input_W=256, max_people=200) :
        super().__init__()
        self.max_people = max_people
        self.features = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        feat_size = input_H//8 * input_W//8 * 128
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_size, 512), nn.ReLU(),
            nn.Linear(512, self.max_people * 2),
        )

    def forward(self, left_img, right_img) :
        x = torch.cat([left_img, right_img], dim=1)
        x = self.features(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.max_people, 2)
        return x


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