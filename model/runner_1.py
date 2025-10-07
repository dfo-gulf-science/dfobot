from solver import run_solver

config_dict = {
    "IMAGE_FOLDER_DIR": "/home/stoyelq/my_hot_storage/dfobot_working/oto_classifier/",
    "NUM_WORKERS": 4,
    "CROP_SIZE": 600,
    "VAL_CROP_SIZE": 600,
    "LEARNING_RATE": 1e-5,
    "BATCH_SIZE": 10,
    "PRINT_EVERY": 25,
    "MAX_DATA": None,
    "NUM_EPOCHS": 3,
    "WEIGHT_DECAY": 0,
    "ACC_SAMPLES": 5,
    "ACC_VAL_SAMPLES": 5,
}

run_solver(device="cuda:1", all_layers=True, config_dict=config_dict)
           # load_checkpoint="/home/stoyelq/my_hot_storage/dfobot_working/run_logs/049__2025-10-06/epochs/epoch_82.pkl")

