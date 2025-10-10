from model.classifier_solver import run_class_solver
from solver import run_solver

dev_count = 1
device = f"cuda:{dev_count}"
learning_rates = [1e-6, 1e-5, 1e-4, 1e-3]
weight_decays = [0, 1e-7, 5e-7, 1e-6, 5e-6]
crop_sizes = [600, 512, 400, 300]
config_dict = {
    "IMAGE_FOLDER_DIR": "/home/stoyelq/my_hot_storage/dfobot_working/oto_classifier/",
    "NUM_WORKERS": 4,
    "VAL_CROP_SIZE": 244,
    "BATCH_SIZE": 20,
    "PRINT_EVERY": 25,
    "MAX_DATA": None, # 150,
    "NUM_EPOCHS": 15,
    "ACC_SAMPLES": 200,
    "ACC_VAL_SAMPLES": 200,
}

acc_history = []
for lr in learning_rates:
    for weight_decay in weight_decays:
        for cs in crop_sizes:
            config_dict["CROP_SIZE"] = cs
            config_dict["LEARNING_RATE"] = lr
            config_dict["WEIGHT_DECAY"] = weight_decay
            solver = run_class_solver(device=device, config_dict=config_dict)
            acc_history.append(f"Learning rate: {lr}, crop size: {cs}, and weight_decay: {weight_decay}. \n Best validation accuracy: {solver.best_val_acc}")

for test_run in acc_history:
    print(test_run)
