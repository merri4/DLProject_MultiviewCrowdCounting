import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from timm import create_model


class MultiViewFusionModel(nn.Module):
    def __init__(self, pretrained_backbone=True):
        super(MultiViewFusionModel, self).__init__()
        
        # 1. Backbone: ConvNeXt-Tiny feature extractor (shared for both left and right images)
        # Load ConvNeXt-Tiny backbone. We remove its classification head to use it as a feature extractor.
        self.backbone = create_model('convnext_tiny', pretrained=True, features_only=True)

        # 2. Fusion module: simple concatenation + convolution to merge left/right features
        # We will fuse at an intermediate feature map resolution (after a certain stage of ConvNeXt).
        # ConvNeXt-Tiny has stages with output dims [96, 192, 384, 768]. We'll fuse after the third stage (dim=384).
        # The fusion conv will take 384*2 channels (left+right) and output 384 channels.
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(384 * 2, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 3. Upsampling path: increase resolution of fused features for finer localization
        # We'll upsample the fused 1/16 resolution feature map to 1/8 resolution for prediction.
        self.upsample_block = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2),  # upsample by 2 (e.g., from 1/16 -> 1/8 scale)
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),         # refine with a conv at 1/8 scale
            nn.ReLU(inplace=True)
        )
        
        # 4. P2PNet-style head: two branches for confidence and coordinate offsets
        # Confidence head: 1 channel per spatial location (sigmoid will be applied later for probability if needed).
        self.conf_head = nn.Conv2d(192, 1, kernel_size=1)
        # Offset head: 2 channels (dx, dy) per spatial location, representing predicted head position offset within the cell.
        self.offset_head = nn.Conv2d(192, 2, kernel_size=1)
        
    def forward(self, left_img, right_img) :
        # Extract features from left and right images using the shared ConvNeXt backbone.
        # We will stop at the intermediate stage (before final downsampling) to get a mid-level feature map.
        # ConvNeXt forward returns classification logits by default, so we manually run the layers to get feature maps.
        # We assume the backbone has attributes 'downsample_layers' and 'stages' as in the original ConvNeXt implementation.
        
        features_left = self.backbone(left_img)     # List of 4 tensors
        features_right = self.backbone(right_img)

        feat_left = features_left[2]   # shape [B, 384, 16, 16]
        feat_right = features_right[2]

        # Now x_left and x_right are the stage-2 feature maps (shape: [B, 384, H/16, W/16]).
        
        # Fuse the left and right features by concatenation along channels
        fused = torch.cat([feat_left, feat_right], dim=1)  # shape: [B, 768, H/16, W/16]
        fused = self.fusion_conv(fused)             # shape: [B, 384, H/16, W/16]
        
        # Upsample to get finer feature map (1/8 resolution) for localization
        features = self.upsample_block(fused)       # shape: [B, 192, H/8, W/8]
        
        # Prediction heads
        conf_map = self.conf_head(features)         # shape: [B, 1, H/8, W/8], raw confidence (logits)
        offset_map = self.offset_head(features)     # shape: [B, 2, H/8, W/8], raw offsets
        
        # (Optionally, apply sigmoid to conf_map to get probabilities and to offset_map to constrain offsets 0-1)
        conf_map = torch.sigmoid(conf_map)  # If using as confidence probability in inference
        offset_map = torch.sigmoid(offset_map)  # If we want offsets normalized between 0 and 1 within each cell
        
        return conf_map, offset_map


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