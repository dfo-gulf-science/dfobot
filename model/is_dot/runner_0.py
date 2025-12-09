from model.classifier_solver import run_class_solver
from model.goodness.goodness_dataloader import get_goodness_dataloader
from model.is_dot.is_dot_dataloader import get_is_dot_dataloader

classes = ["No", "Dot"]

config_dict = {
    "IMAGE_FOLDER_DIR": "/home/stoyelq/my_hot_storage/dfobot_working/dots/",
    "get_dataloaders": get_is_dot_dataloader,
    "NUM_WORKERS": 4,
    "CROP_SIZE": 300,
    "VAL_CROP_SIZE": 300,
    "LEARNING_RATE": 1e-4,
    "BATCH_SIZE": 10,
    "PRINT_EVERY": 100,
    "MAX_DATA": 500,
    "NUM_EPOCHS": 10,
    "WEIGHT_DECAY": 5e-7,
    "ACC_SAMPLES": 100,
    "ACC_VAL_SAMPLES": 100,
}

run_class_solver(device="cuda:0", config_dict=config_dict, classes=classes)
           # load_checkpoint="/home/stoyelq/my_hot_storage/dfobot_working/run_logs/049__2025-10-06/epochs/epoch_82.pkl")

