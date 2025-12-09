from aging_solver import run_aging_solver
from model.ages.aging_dataloader import get_aging_dataloaders
from model.centers.centers_dataloader import get_center_dataloaders

config_dict = {
    "IMAGE_FOLDER_DIR": "/home/stoyelq/my_hot_storage/dfobot_working/ages_combined/",
    "get_dataloaders": get_aging_dataloaders,
    "NUM_WORKERS": 4,
    "CROP_SIZE": 500,
    "VAL_CROP_SIZE": 500,
    "LEARNING_RATE": 1e-4,
    "BATCH_SIZE": 20,
    "PRINT_EVERY": 100,
    "MAX_DATA": None,
    "NUM_EPOCHS": 5,
    "WEIGHT_DECAY": 1e-5,
    "ACC_SAMPLES": 500,
    "ACC_VAL_SAMPLES": 500,
}

run_aging_solver(device="cuda:0", config_dict=config_dict)
           # load_checkpoint="/home/stoyelq/my_hot_storage/dfobot_working/run_logs/049__2025-10-06/epochs/epoch_82.pkl")

