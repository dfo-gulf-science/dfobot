import os
import tkinter as tk
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import model.solver as solver
from model.model_utils import ClassifierModel

IMAGE_DIR = "/home/stoyelq/my_hot_storage/dfobot_working/centers/train"
# IMAGE_DIR = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/raw"
WEIGHTS_PATH = "/home/stoyelq/my_hot_storage/dfobot_working/run_logs/095__2025-10-14/trained_weights.pth"
DEVICE = "cuda:1"


class CenterChecker:
    def __init__(self, image_size=(224, 224)):
        model = ClassifierModel(num_outputs=2)
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
        display_img = pil_img.resize((224, 224))

        self.tk_img = ImageTk.PhotoImage(display_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            print(output)
            x0 = int(output[0][0] * self.image_size[0])
            x1 = int(x0 + 10)
            y0 = int(output[0][1] * self.image_size[1])
            y1 = int(y0 + 10)
            self.canvas.create_line(x0,y0, x1, y1, arrow=tk.FIRST, width=3, fill="red")

    def next_image(self, event=None):
        self.index = (self.index + 1) % len(self.image_paths)
        self.display_image()

    def prev_image(self, event=None):
        self.index = (self.index - 1) % len(self.image_paths)
        self.display_image()

CenterChecker()