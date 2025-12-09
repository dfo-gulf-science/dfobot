from model.ref_lines.ref_line_dataloader import get_ref_line_dataloaders
from model.ref_lines.ref_line_solver import run_ref_line_solver

config_dict = {
    "IMAGE_FOLDER_DIR": "/home/stoyelq/my_hot_storage/dfobot_working/ref_line/",
    "get_dataloaders": get_ref_line_dataloaders,
    "NUM_WORKERS": 4,
    "CROP_SIZE": 500,
    "VAL_CROP_SIZE": 500,
    "LEARNING_RATE": 1e-5,
    "BATCH_SIZE": 10,
    "PRINT_EVERY": 25,
    "MAX_DATA": None,
    "NUM_EPOCHS": 10,
    "WEIGHT_DECAY": 1e-5,
    "ACC_SAMPLES": 200,
    "ACC_VAL_SAMPLES": 200,
}

run_ref_line_solver(device="cuda:0", config_dict=config_dict)
           # load_checkpoint="/home/stoyelq/my_hot_storage/dfobot_working/run_logs/049__2025-10-06/epochs/epoch_82.pkl")

