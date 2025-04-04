import gradio as gr
from model import *

def predict(image1, image2):
    density_map, count = MultiviewFusionModel(image1, image2)
    return density_map, count

if __name__ == "__main__" :
    
    demo = gr.Interface(
        fn=predict,
        inputs=[gr.Image(label="View 1"), gr.Image(label="View 2")],
        outputs=[gr.Image(label="Density Map"), gr.Number(label="Count")],
        title="Multi-View Crowd Counter"
    )
    
    demo.launch()