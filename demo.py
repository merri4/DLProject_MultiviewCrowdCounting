import gradio as gr
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

import cv2

from model import MultiViewFusionModel
import matplotlib.pyplot as plt

def load_checkpoint(filename, model, optimizer=None):  
    
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])  
    
    if optimizer is not None :
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print("Model {} Loaded! Loss {} on Epoch {}".format(filename, loss, epoch))

    return epoch, loss


# 모델 로딩해놓기
MODEL_PATH = "./output_p2p/epoch_3.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiViewFusionModel().to(DEVICE)
load_checkpoint(MODEL_PATH, model)


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

def predict(image1, image2, threshold) :
    
    # 이미지 전처리
    print(image1.shape)
    print(image2.shape)

    left_img = Image.fromarray(image1).convert('RGB').resize((W, H))
    right_img = Image.fromarray(image2).convert('RGB').resize((W, H))
    
    to_tensor = T.ToTensor()
    left_img = to_tensor(left_img).unsqueeze(0).to(DEVICE)
    right_img = to_tensor(right_img).unsqueeze(0).to(DEVICE)

    print(left_img.shape)
    print(right_img.shape)

    # 모델 feeding
    with torch.no_grad() :
        conf_map, offset_map = model(left_img, right_img)
    
    conf_map = torch.sigmoid(conf_map[0, 0])  # shape: [Hf, Wf]
    offset_map = offset_map[0]               # shape: [2, Hf, Wf]
    dx, dy = offset_map[0], offset_map[1]    # each: [Hf, Wf]

    Hf, Wf = conf_map.shape
    cell_h, cell_w = H / Hf, W / Wf

    y_grid, x_grid = torch.meshgrid(torch.arange(Hf), torch.arange(Wf), indexing='ij')
    x_grid = x_grid.float().to(DEVICE)
    y_grid = y_grid.float().to(DEVICE)

    pred_x = (x_grid + 0.5) * cell_w + dx * cell_w
    pred_y = (y_grid + 0.5) * cell_h + dy * cell_h

    # Flatten and threshold by confidence
    conf_flat = conf_map.view(-1)
    x_flat = pred_x.view(-1)
    y_flat = pred_y.view(-1)

    keep = conf_flat > threshold
    pred_coords = torch.stack([x_flat[keep], y_flat[keep]], dim=1).cpu().numpy()

    # Normalize coordinates to [0, 1] range for coords_to_density_map
    pred_coords_norm = pred_coords / np.array([[W, H]])

    pred_density = coords_to_density_map(pred_coords_norm, H, W, sigma=6)
    count = len(pred_coords)
    print(count)

    density_map = plt.figure()
    ax = density_map.add_subplot(1,1,1)
    ax.imshow(pred_density, cmap='jet', origin='lower')
    ax.set_title(f"Pred Density, People: {count}")
    ax.axis('off')

    density_map.tight_layout()

    return density_map, count

if __name__ == "__main__" :

    H, W = 256, 256
    MAX_NUMBER = 200
    
    demo = gr.Interface(
        fn=predict,
        inputs=[gr.Image(label="View 1"), gr.Image(label="View 2"), gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.8)],
        outputs=[gr.Plot(label="Density Map"), gr.Number(label="Count")],
        title="Multi-View Crowd Counter",
        description="This demo estimates the crowd count from two different views of the same scene."
    )
    
    demo.launch()