import os
import tkinter as tk

import pandas as pd
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import model.solver as solver
from model.model_utils import ClassifierModel, AugmentedModel, load_model_from_log_file

IMAGE_DIR = "/home/stoyelq/my_hot_storage/dfobot_working/ages/val/"
# IMAGE_DIR = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/raw"
WEIGHTS_PATH = "/home/stoyelq/my_hot_storage/dfobot_working/run_logs/010__2025-11-13/trained_weights.pth"
WIDTH_WEIGHTS_PATH = "/home/stoyelq/my_hot_storage/dfobot_working/run_logs/006__2025-11-13/trained_weights.pth"
BOTH_WEIGHTS_PATH = "/home/stoyelq/Desktop/work/dfobot/results/edge_type/trained_weights.pth"
# WEIGHTS_PATH = "/home/stoyelq/Desktop/work/dfobot/results/goodness/trained_weights.pth"
# LOG_PATH = "/home/stoyelq/my_hot_storage/dfobot_working/run_logs/056__2025-11-03"
# LOG_PATH = "/home/stoyelq/my_hot_storage/dfobot_working/run_logs/062__2025-11-04"
DEVICE = "cuda:0"
# labels = ["Good", "Crack", "Crystal", "Twin"]
# LABELS = ["No dot", "Dot"]
LABELS = ["Hyaline", "Opaque"]
WIDTH_LABELS = ["Narrow", "Wide"]
BOTH_LABELS = ["foo", "Narrow Hyaline", "Wide Hyaline", "Narrow Opaque", "Wide Opaque", "foo", "foo", "foo"]


def get_age_from_path(path):
    metadata_df = pd.read_csv("/home/stoyelq/my_hot_storage/dfobot_working/ages/ages.csv")
    # make sure this function returns the label from the path
    uuid = path.split(".")[0]
    metadata_row = metadata_df[(metadata_df["uuid"] == uuid)].iloc[0]
    result = torch.tensor([float(metadata_row["width"])])
    width_result =WIDTH_LABELS[metadata_row["width"]]
    type_result =LABELS[metadata_row["edge_trans"]]
    return width_result, type_result

class ImageChecker:
    def __init__(self, image_size=(500, 500)):
        model = ClassifierModel(num_outputs=len(LABELS))
        # model = ClassifierModel(num_outputs=2)
        # model = AugmentedModel(num_outputs=1, metadata_length=1)
        model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))

        width_model = ClassifierModel(num_outputs=len(WIDTH_LABELS))
        width_model.load_state_dict(torch.load(WIDTH_WEIGHTS_PATH, weights_only=True))
        both_model = ClassifierModel(num_outputs=len(BOTH_LABELS))
        both_model.load_state_dict(torch.load(BOTH_WEIGHTS_PATH, weights_only=True))


        # model = load_model_from_log_file(LOG_PATH)

        self.model = model.eval()
        self.width_model = width_model.eval()
        self.both_model = both_model.eval()
        self.image_dir = IMAGE_DIR
        self.image_size = image_size
        self.device = torch.device(DEVICE)
        self.model.to(self.device)
        self.width_model.to(self.device)
        self.both_model.to(self.device)

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
        self.both_label = tk.Label(self.root, text="", font=("Arial", 16))
        self.both_label.pack()

        self.root.bind("<Right>", self.next_image)
        self.root.bind("<Left>", self.prev_image)

        self.display_image()

        self.root.mainloop()

    def display_image(self):
        img_path = self.image_paths[self.index]
        pil_img = Image.open(img_path).convert("RGB")
        display_img = pil_img.resize((500, 500))
        tk_img = ImageTk.PhotoImage(display_img)
        self.canvas.config(image=tk_img)
        self.canvas.image = tk_img

        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with (torch.no_grad()):
            output = self.model(input_tensor)
            width_output = self.width_model(input_tensor)
            both_output = self.both_model(input_tensor)

            prediction = LABELS[int(torch.max(output[0], 0)[1].item())]
            width_prediction = WIDTH_LABELS[int(torch.max(width_output[0], 0)[1].item())]
            both_prediction = BOTH_LABELS[int(torch.max(both_output[0], 0)[1].item())]

            width_confidence = torch.abs(torch.diff(output[0]))
            trans_confidence = torch.abs(torch.diff(width_output[0]))

            both_width_confidence = torch.abs(torch.diff(both_output[0][1:3])) + torch.abs(torch.diff(both_output[0][3:5]))
            both_trans_confidence = torch.abs(both_output[0][1] - both_output[0][3]) + torch.abs(both_output[0][2] - both_output[0][4])
            print(both_output)
            print(f"Width: {width_confidence / both_width_confidence}%")
            print(f"Trans: {trans_confidence / both_trans_confidence}%")
            # prediction = output[0][0]

        real_value, type_result = get_age_from_path(img_path.split("/")[-1])

        self.real_label.config(text=f"Real Value: {real_value},  {type_result}")
        self.pred_label.config(text=f"Predicted Value: {width_prediction}, {prediction}")
        self.both_label.config(text=f"Predicted Values: {both_prediction}")

    def next_image(self, event=None):
        self.index = (self.index + 1) % len(self.image_paths)
        self.display_image()

    def prev_image(self, event=None):
        self.index = (self.index - 1) % len(self.image_paths)
        self.display_image()

ImageChecker()
