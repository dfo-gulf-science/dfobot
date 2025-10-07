from solver import run_solver

config_dict = {
    "IMAGE_FOLDER_DIR": "/home/stoyelq/my_hot_storage/dfobot_working/crack_finder/",
    "NUM_WORKERS": 4,
    "CROP_SIZE": 500,
    "VAL_CROP_SIZE": 500,
    "LEARNING_RATE": 1e-6,
    "BATCH_SIZE": 20,
    "PRINT_EVERY": 25,
    "MAX_DATA":  None,
    "NUM_EPOCHS": 10,
    "WEIGHT_DECAY": 1e-7,
    "ACC_SAMPLES": 200,
    "ACC_VAL_SAMPLES": 200,
}

run_solver(device="cuda:1", all_layers=True, config_dict=config_dict)
           # load_checkpoint="/home/stoyelq/Documents/dfobot_data/gpu_1/epoch_1.pkl")

