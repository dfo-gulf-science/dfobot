from classifier_solver import run_class_solver

config_dict = {
    "IMAGE_FOLDER_DIR": "/home/stoyelq/my_hot_storage/dfobot_working/oto_classifier/",
    "NUM_WORKERS": 4,
    "CROP_SIZE": 224,
    "VAL_CROP_SIZE": 224,
    "LEARNING_RATE": 1e-5,
    "BATCH_SIZE": 10,
    "PRINT_EVERY": 25,
    "MAX_DATA": None,
    "NUM_EPOCHS": 10,
    "WEIGHT_DECAY": 0,
    "ACC_SAMPLES": 100,
    "ACC_VAL_SAMPLES": 100,
}

run_class_solver(device="cuda:0", config_dict=config_dict)
           # load_checkpoint="/home/stoyelq/my_hot_storage/dfobot_working/run_logs/049__2025-10-06/epochs/epoch_82.pkl")

