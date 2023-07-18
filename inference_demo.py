import torch
import numpy as np
from torchvision import transforms
import gradio as gr






transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])
model = load_model()
model.to('cuda')
model.eval()


def inference(image):
    image = transform(image)
    output, _ = model(image, mode='sample')
    return output


demo = gr.Interface(
    inference,
    gr.Image(type="pil"),
    gr.Text()
)

if __name__ == "__main__":
    demo.launch()
