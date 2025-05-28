import gradio as gr
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

import cv2

from model import MultiviewFusionModel
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

# 모델 불러와두기
MODEL_PATH = "./output_rescon/epoch_3.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiviewFusionModel().to(DEVICE)
load_checkpoint(MODEL_PATH, model)


def coords_to_density_map(output, H, W, sigma=6, threshold=0.5) :

    output = output.detach().cpu().numpy()

    conf = output[:, :, 2].squeeze(0)       # [N, 1]
    coords = output[:, :, :2].squeeze(0)    # [N, 2]

    true_coords = []
    for i,c in enumerate(coords) :
        if conf[i] > threshold :
            true_coords.append(c)
    
    count = len(true_coords)

    density = np.zeros((H, W), dtype=np.float32)
    
    for x, y in true_coords :
        x_pix = int(np.clip(x, 0, 1) * (W - 1))
        y_pix = int(np.clip(y, 0, 1) * (H - 1))
        if 0 <= x_pix < W and 0 <= y_pix < H:
            density[y_pix, x_pix] = 1
    
    density = cv2.GaussianBlur(density, (15, 15), sigma)
    
    if density.sum() > 0:
        density = density * (len(true_coords) / density.sum())
    
    return density, count

def predict(image1, image2, threshold) :
    
    # 이미지 전처리
    left_img = Image.fromarray(image1).convert('RGB').resize((W, H))
    right_img = Image.fromarray(image2).convert('RGB').resize((W, H))
    
    to_tensor = T.ToTensor()
    left_img = to_tensor(left_img).unsqueeze(0).to(DEVICE)
    right_img = to_tensor(right_img).unsqueeze(0).to(DEVICE)

    # 모델 feeding
    with torch.no_grad() :
        output = model(left_img, right_img)
    
    pred_density, count = coords_to_density_map(output, H, W, sigma=6, threshold=threshold)

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
        inputs=[gr.Image(label="View 1"), gr.Image(label="View 2"), gr.Slider(label="Confidence", minimum=0, maximum=1.0, value=0.5, step=0.05)],
        outputs=[gr.Plot(label="Density Map"), gr.Number(label="Count")],
        title="Multi-View Crowd Counter",
        description="This demo estimates the crowd count from two different views of the same scene."
    )
    
    demo.launch()