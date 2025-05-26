import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

class MultiviewFusionModel(nn.Module):
    def __init__(self, input_H=256, input_W=256, max_people=200):
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

    def forward(self, left_img, right_img):
        x = torch.cat([left_img, right_img], dim=1)
        x = self.features(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.max_people, 2)
        return x

if __name__ == "__main__" :
    pass