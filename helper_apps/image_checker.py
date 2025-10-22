import os
import tkinter as tk
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import model.solver as solver
from model.model_utils import ClassifierModel

# IMAGE_DIR = "/home/stoyelq/my_hot_storage/dfobot_working/crack_finder/train/"
IMAGE_DIR = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/raw"
WEIGHTS_PATH = "/home/stoyelq/my_hot_storage/dfobot_working/run_logs/152__2025-10-20/trained_weights.pth"
DEVICE = "cuda:0"
# labels = ["Good", "Crack", "Crystal", "Twin"]
LABELS = ["Good", "Bad"]


class ImageChecker:
    def __init__(self, image_size=(224, 224)):
        model = ClassifierModel(num_outputs=len(LABELS))
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
        self.root.title("Regression Model Viewer")
        self.canvas = tk.Label(self.root)
        self.canvas.pack()
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
            print(img_path)
            prediction = LABELS[int(torch.max(output[0], 0)[1].item())]


        self.pred_label.config(text=f"Predicted Value: {prediction}")

    def next_image(self, event=None):
        self.index = (self.index + 1) % len(self.image_paths)
        self.display_image()

    def prev_image(self, event=None):
        self.index = (self.index - 1) % len(self.image_paths)
        self.display_image()

ImageChecker()