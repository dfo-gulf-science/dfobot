from model.classifier_solver import run_class_solver
from model.edge_type.edge_type_dataloader import get_edge_type_dataloader

classes = ["Hyaline", "Opaque"]
# classes = ["Narrow", "Wide"]

config_dict = {
    "IMAGE_FOLDER_DIR": "/home/stoyelq/my_hot_storage/dfobot_working/ages/",
    "get_dataloaders": get_edge_type_dataloader,
    "COL_NAME": "edge_trans",
    # "COL_NAME": "width",
    "NUM_WORKERS": 4,
    "CROP_SIZE": 500,
    "VAL_CROP_SIZE": 500,
    "LEARNING_RATE": 1e-4,
    "BATCH_SIZE": 10,
    "PRINT_EVERY": 100,
    "MAX_DATA": None,
    "NUM_EPOCHS": 20,
    "WEIGHT_DECAY": 0,
    "ACC_SAMPLES": 200,
    "ACC_VAL_SAMPLES": 200,
}

run_class_solver(device="cuda:0", config_dict=config_dict, classes=classes)
           # load_checkpoint="/home/stoyelq/my_hot_storage/dfobot_working/run_logs/049__2025-10-06/epochs/epoch_82.pkl")

