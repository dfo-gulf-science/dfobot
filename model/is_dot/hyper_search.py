from datetime import datetime

from model.centers.center_solver import run_center_solver
from model.classifier_solver import run_class_solver
import os
import csv

from model.goodness.goodness_dataloader import get_goodness_dataloader
from model.is_dot.is_dot_dataloader import get_is_dot_dataloader

LOG_DIR= "/home/stoyelq/my_hot_storage/dfobot_working/run_logs/"


dev_count = 0
device = f"cuda:{dev_count}"
learning_rates = [1e-6, 1e-5, 1e-4, 1e-3]
weight_decays = [0, 1e-7, 5e-7, 1e-6, 5e-6]
crop_sizes = [300]
LABELS = ["No", "Dot"]

config_dict = {
    "IMAGE_FOLDER_DIR": "/home/stoyelq/my_hot_storage/dfobot_working/dots/",
    "get_dataloaders": get_is_dot_dataloader,
    "NUM_WORKERS": 4,
    "VAL_CROP_SIZE": 300,
    "BATCH_SIZE": 20,
    "PRINT_EVERY": 100,
    "MAX_DATA": None, # 150,
    "NUM_EPOCHS": 20,
    "LOG_EPOCHS": False,
    "ACC_SAMPLES": 200,
    "ACC_VAL_SAMPLES": 200,
}


def get_hyper_log_dir():
    try:
        max_count = max([int(file_name[4:7])for file_name in os.listdir(LOG_DIR) if "hyper" in file_name])
        run_count = str(max_count + 1).zfill(3)
    except ValueError:
        run_count = '002'
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
