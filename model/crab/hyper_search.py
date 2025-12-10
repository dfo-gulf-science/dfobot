from datetime import datetime

from model.centers.center_solver import run_center_solver
from model.classifier_solver import run_class_solver
import os
import csv

from model.crab.crab_dataloader import get_crab_dataloader
from model.edge_type.edge_type_dataloader import get_edge_type_dataloader

LOG_DIR= "/home/stoyelq/my_hot_storage/dfobot_working/run_logs/"


dev_count = 1
device = f"cuda:{dev_count}"
learning_rates = [2e-4, 1e-4, 5e-3]
weight_decays = [0, 5e-6]
crop_sizes = [500]
LABELS = ["0", "1", "2", "3", "4", "5", "6", "7"]


config_dict = {
    "IMAGE_FOLDER_DIR": "/home/stoyelq/my_hot_storage/dfobot_working/ages/",
    "get_dataloaders": get_crab_dataloader,
    "COL_NAME": "tow_type",
    "NUM_WORKERS": 4,
    "VAL_CROP_SIZE": 500,
    "BATCH_SIZE": 20,
    "PRINT_EVERY": 200,
    "MAX_DATA": None, # 150,
    "NUM_EPOCHS": 40,
    "LOG_EPOCHS": False,
    "ACC_SAMPLES": 200,
    "ACC_VAL_SAMPLES": 200,
}


def get_hyper_log_dir():
    try:
        max_count = max([int(file_name[7:10])for file_name in os.listdir(LOG_DIR) if "hyper" in file_name])
        run_count = str(max_count + 1).zfill(3)
    except ValueError:
        run_count = '001'
    run_log_dir_name = f"hyper__{run_count}__{datetime.today().strftime('%Y-%m-%d')}"
    run_log_dir_path = os.path.join(LOG_DIR, run_log_dir_name)
    epochs_log_dir_path = os.path.join(run_log_dir_path, 'epochs')
    os.makedirs(run_log_dir_path, exist_ok=True)
    os.makedirs(epochs_log_dir_path, exist_ok=True)
    return run_log_dir_path

# set up log:
acc_history = []
hyper_log_path = os.path.join(get_hyper_log_dir(), "log.csv")
with open(hyper_log_path,'w') as hyper_log:
    config_file_writer = csv.writer(hyper_log)
    config_file_writer.writerow(["learning_rate", "weight_decay", "crop_size", "accuracy"])

# Loop it!
for lr in learning_rates:
    for weight_decay in weight_decays:
        for cs in crop_sizes:
            config_dict["CROP_SIZE"] = cs
            config_dict["LEARNING_RATE"] = lr
            config_dict["WEIGHT_DECAY"] = weight_decay
            solver = run_class_solver(device=device, config_dict=config_dict, classes=LABELS)
            with open(hyper_log_path, "a") as hyper_log:
                hyper_log.write(f"{lr}, {weight_decay}, {cs}, {max(solver.test_acc_history)}\n")

            acc_history.append(f"Learning rate: {lr}, crop size: {cs}, and weight_decay: {weight_decay}. \n Best validation accuracy: {max(solver.test_acc_history)}")

for test_run in acc_history:
    print(test_run)
