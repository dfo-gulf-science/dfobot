import os
import shutil
import tkinter as tk
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import model.solver as solver
from model.model_utils import ClassifierModel

# IMAGE_DIR = "/home/stoyelq/my_hot_storage/dfobot_working/crack_finder/train/"
IMAGE_DIR = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/raw"
OUT_DIR = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/sorted"
WEIGHTS_PATH = "/home/stoyelq/my_hot_storage/dfobot_working/run_logs/057__2025-10-08/trained_weights.pth"
DEVICE = "cuda:1"
labels = ["Good", "Crack", "Crystal", "Twin"]

for label in labels:
    label_dir = os.path.join(OUT_DIR, label)
    os.makedirs(label_dir, exist_ok=True)


model = ClassifierModel(num_outputs=len(labels))
model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))

# model = torch.load(MODEL_PATH, weights_only=False)
model = model.eval()
device = torch.device(DEVICE)
model.to(device)

image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
index = 0

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

count = 0
for image_filename in os.listdir(IMAGE_DIR):
    count += 1
    if count % 1000 == 0:
        print(count)
    image_path = os.path.join(IMAGE_DIR, image_filename)
    pil_img = Image.open(image_path).convert("RGB")

    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = labels[int(torch.max(output[0], 0)[1].item())]

        dst = os.path.join(OUT_DIR, prediction, image_filename)
        shutil.copy(image_path, dst)
