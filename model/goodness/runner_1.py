from model.classifier_solver import run_class_solver
from model.goodness.goodness_dataloader import get_goodness_dataloader

classes = ["Good", "Bad"]

config_dict = {
    "IMAGE_FOLDER_DIR": "/home/stoyelq/my_hot_storage/dfobot_working/goodness/",
    "get_dataloaders": get_goodness_dataloader,
    "NUM_WORKERS": 4,
    "CROP_SIZE": 300,
    "VAL_CROP_SIZE": 224,
    "LEARNING_RATE": 5e-6,
    "BATCH_SIZE": 10,
    "PRINT_EVERY": 40,
    "MAX_DATA": None,
    "NUM_EPOCHS": 40,
    "WEIGHT_DECAY": 0,
    "ACC_SAMPLES": 100,
    "ACC_VAL_SAMPLES": 100,
}

run_class_solver(device="cuda:0", config_dict=config_dict, classes=classes)
           # load_checkpoint="/home/stoyelq/my_hot_storage/dfobot_working/run_logs/049__2025-10-06/epochs/epoch_82.pkl")

