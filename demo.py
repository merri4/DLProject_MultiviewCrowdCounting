import gradio as gr
from PIL import Image
from model import *
import numpy as np


def predict(image1, image2) :
    
    # 이미지 전처리

    density_map, count = MultiviewFusionModel(image1, image2)

    # 샘플 작동
    # sample_density_map = Image.open("./src/sample_gt.png")
    # sample_count = 42  # Example hardcoded count

    return density_map, count

if __name__ == "__main__" :
    
    demo = gr.Interface(
        fn=predict,
        inputs=[gr.Image(label="View 1"), gr.Image(label="View 2")],
        outputs=[gr.Image(label="Density Map"), gr.Number(label="Count")],
        title="Multi-View Crowd Counter",
        description="This demo estimates the crowd count from two different views of the same scene."
    )
    
    demo.launch()