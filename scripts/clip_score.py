from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torchmetrics.multimodal.clip_score import CLIPScore
from functools import partial

output_path = "./outputs/txt2img-samples/2024-04-19-22-19-57"

# Load the image
image_path = "/samples/00000.png"
image = Image.open(output_path+image_path)

# Preprocess the image
transform = transforms.Compose([
    transforms.ToTensor(),
])

tensor = transform(image).unsqueeze(0)

metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")
score = metric(tensor, "a puppy wearing a hat")
print(score.detach())