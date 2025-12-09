from datetime import datetime

from model.centers.center_solver import run_center_solver
from model.classifier_solver import run_class_solver
import os
import csv
from model.edge_type.edge_type_dataloader import get_edge_type_dataloader
from model.ref_lines.ref_line_dataloader import get_ref_line_dataloaders
from model.ref_lines.ref_line_solver import run_ref_line_solver

LOG_DIR= "/home/stoyelq/my_hot_storage/dfobot_working/run_logs/"


dev_count = 1
device = f"cuda:{dev_count}"
learning_rates = [1e-4, 1e-3, 5e-4]
weight_decays = [0, 1e-5]
LABELS = ["Narrow", "Wide"]


config_dict = {
    "IMAGE_FOLDER_DIR": "/home/stoyelq/my_hot_storage/dfobot_working/ref_line/",
    "get_dataloaders": get_ref_line_dataloaders,
    "NUM_WORKERS": 4,
    "CROP_SIZE": 500,
    "VAL_CROP_SIZE": 500,
    "BATCH_SIZE": 20,
    "PRINT_EVERY": 200,
    "MAX_DATA": None, # 150,
    "NUM_EPOCHS": 50,
    "LOG_EPOCHS": False,
    "ACC_SAMPLES": 200,
    "ACC_VAL_SAMPLES": 200,
}


def get_hyper_log_dir():
    try:
        max_count = max([int(file_name[7:10])for file_name in os.listdir(LOG_DIR) if "hyper" in file_name])
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
        config_dict["LEARNING_RATE"] = lr
        config_dict["WEIGHT_DECAY"] = weight_decay
        solver = run_ref_line_solver(device=device, config_dict=config_dict)
        with open(hyper_log_path, "a") as hyper_log:
            hyper_log.write(f"{lr}, {weight_decay}, {max(solver.test_acc_history)}\n")

        acc_history.append(f"Learning rate: {lr},, and weight_decay: {weight_decay}. \n Best validation accuracy: {max(solver.test_acc_history)}")

for test_run in acc_history:
    print(test_run)
