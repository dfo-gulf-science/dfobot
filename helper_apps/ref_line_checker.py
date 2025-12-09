import os
import tkinter as tk

import pandas as pd
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import model.solver as solver
from model.model_utils import ClassifierModel

# IMAGE_DIR = "/home/stoyelq/my_hot_storage/dfobot_working/centers/train"
IMAGE_DIR = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/ref_line"
# IMAGE_DIR = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/goodness"
WEIGHTS_PATH = "/home/stoyelq/my_hot_storage/dfobot_working/run_logs/050__2025-11-19/trained_weights.pth"

DEVICE = "cuda:1"

IMAGE_SIZE = 500
class RefLineChecker:
    def __init__(self, image_size=(IMAGE_SIZE, IMAGE_SIZE)):
        model = ClassifierModel(num_outputs=4)
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
        self.root.title("Center Viewer")

        self.canvas = tk.Canvas(self.root, width=self.image_size[0], height=self.image_size[1])
        self.canvas.pack()

        self.tk_image = None

        self.root.bind("<Right>", self.next_image)
        self.root.bind("<Left>", self.prev_image)

        self.display_image()

        self.root.mainloop()

    def display_image(self):
        self.canvas.delete("all")
        img_path = self.image_paths[self.index]
        pil_img = Image.open(img_path).convert("RGB")
        display_img = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE))

        self.tk_img = ImageTk.PhotoImage(display_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        center_x, center_y, edge_x, edge_y = get_line_from_path(img_path)

        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            print(output)
            x0 = int(output[0][0] * 500)
            y0 = int(output[0][1] * 500)
            x1 = int(output[0][2] * 500)
            y1 = int(output[0][3] * 500)

        self.canvas.create_line(center_x / 2,center_y / 2, edge_x / 2 , edge_y / 2, arrow=tk.FIRST, width=3, fill="red")
        self.canvas.create_line(x0, y0, x1, y1, arrow=tk.FIRST, width=3, fill="blue")

    def next_image(self, event=None):
        self.index = (self.index + 1) % len(self.image_paths)
        self.display_image()

    def prev_image(self, event=None):
        self.index = (self.index - 1) % len(self.image_paths)
        self.display_image()


def get_line_from_path(path):
    metadata_df = pd.read_csv("/home/stoyelq/my_hot_storage/dfobot/yellowtail/ref_line.csv")
    # make sure this function returns the label from the path
    uuid = path.split(".")[0].split("/")[-1]
    try:
        metadata_row = metadata_df[(metadata_df["uuid"] == uuid)].iloc[0]
        return metadata_row["center_x"], metadata_row["center_y"], metadata_row["edge_x"], metadata_row["edge_y"]
    except IndexError:
        return 0, 0, 0, 0

RefLineChecker()