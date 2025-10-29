from model.ages.aging_dataloader import get_aging_dataloaders
from model.ages.aging_solver import AgingSolver
from model.model_utils import ClassifierModel
from model.solver import run_solver, make_solver_plots, make_bot_plot
import torch
import gc

WEIGHTS_PATH = "/home/stoyelq/my_hot_storage/dfobot_working/run_logs/014__2025-10-28/trained_weights.pth"
device = "cuda:1"
def get_dfobot():
    config_dict = {
        'NUM_WORKERS': 1,
        'CROP_SIZE': 500,
        'VAL_CROP_SIZE': 500,
        'IMAGE_FOLDER_DIR': '/home/stoyelq/my_hot_storage/dfobot_working/ages/',
    }
    bot = ClassifierModel(num_outputs=1)
    bot.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))
    data_loaders = get_aging_dataloaders(batch_size=5, max_size=None, config_dict=config_dict)[0]
    return bot, data_loaders["val"]

def get_next_image_prediction(dataloader, bot, device):
    images, data, labels, uuids = next(iter(dataloader))
    images = images.to(device)
    data = data.to(device)
    labels = labels.to(device)
    bot = bot.to(device)
    output = bot(images, data)
    return images, output, labels

bot, dataloader = get_dfobot()
imgs, outputs, labels = get_next_image_prediction(dataloader, bot, device)

import matplotlib.pyplot as plt
# plt.imshow(imgs[0].cpu().permute(1, 2, 0))
# plt.show()

y_pred, y_true = make_bot_plot(bot, 100, dataloader, device, title=str(WEIGHTS_PATH.split("/")[-2]))
