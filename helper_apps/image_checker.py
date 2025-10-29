import os
import tkinter as tk

import pandas as pd
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import model.solver as solver
from model.model_utils import ClassifierModel, AugmentedModel

IMAGE_DIR = "/home/stoyelq/my_hot_storage/dfobot_working/ages/val/"
# IMAGE_DIR = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/raw"
WEIGHTS_PATH = "/home/stoyelq/my_hot_storage/dfobot_working/run_logs/049__2025-10-29/trained_weights.pth"
DEVICE = "cuda:0"
# labels = ["Good", "Crack", "Crystal", "Twin"]
LABELS = ["Good", "Bad"]

def get_age_from_path(path):
    metadata_df = pd.read_csv("/home/stoyelq/my_hot_storage/dfobot_working/ages/ages.csv")
    # make sure this function returns the label from the path
    uuid = path.split(".")[0]
    metadata_row = metadata_df[(metadata_df["uuid"] == uuid)].iloc[0]
    result = torch.tensor([float(metadata_row["annuli"])])
    return result

class ImageChecker:
    def __init__(self, image_size=(500, 500)):
        # model = ClassifierModel(num_outputs=len(LABELS))
        model = ClassifierModel(num_outputs=1)
        model = AugmentedModel(num_outputs=1, metadata_length=1)
        model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))

        self.model = model.eval()
        self.image_dir = IMAGE_DIR
        self.image_size = image_size
        self.device = torch.device(DEVICE)
        self.model.to(self.device)

        self.image_paths = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.index = 0

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

        self.root = tk.Tk()
        self.root.title("Aging Model Viewer")
        self.canvas = tk.Label(self.root)
        self.canvas.pack()
        self.real_label = tk.Label(self.root, text="", font=("Arial", 16))
        self.real_label.pack()
        self.pred_label = tk.Label(self.root, text="", font=("Arial", 16))
        self.pred_label.pack()

        self.root.bind("<Right>", self.next_image)
        self.root.bind("<Left>", self.prev_image)

        self.display_image()

        self.root.mainloop()

    def display_image(self):
        img_path = self.image_paths[self.index]
        pil_img = Image.open(img_path).convert("RGB")
        display_img = pil_img.resize((512, 512))
        tk_img = ImageTk.PhotoImage(display_img)
        self.canvas.config(image=tk_img)
        self.canvas.image = tk_img

        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            print(output)
            # prediction = LABELS[int(torch.max(output[0], 0)[1].item())]
            prediction = output[0][0]

        real_value = get_age_from_path(img_path.split("/")[-1])[0]

        self.real_label.config(text=f"Real Value: {real_value}")
        self.pred_label.config(text=f"Predicted Value: {prediction}")

    def next_image(self, event=None):
        self.index = (self.index + 1) % len(self.image_paths)
        self.display_image()

    def prev_image(self, event=None):
        self.index = (self.index - 1) % len(self.image_paths)
        self.display_image()

ImageChecker()
