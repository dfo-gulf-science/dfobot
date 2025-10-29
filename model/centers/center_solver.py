import csv
import time

import torch
from torch import nn, optim
import os
from model.model_utils import get_center_model
from model.centers.centers_dataloader import get_center_dataloaders
from model.classifier_solver import ClassifierSolver

from model.solver import get_run_log_dir


class CenterSolver(ClassifierSolver):

    def __init__(self, model, criterion, optimizer, config_dict, **kwargs):
        super().__init__(model, criterion, optimizer, config_dict, **kwargs)
        dataloaders, dataset_sizes = get_center_dataloaders(self.batch_size, self.max_data, config_dict=config_dict)
        self.test_dataloader = dataloaders["val"]
        self.train_dataloader = dataloaders["train"]


    def train(self):
        start_time = time.time()
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            num_train = self.dataset_sizes["train"]
            if self.max_data:
                num_train = self.max_data
            iterations_per_epoch = max(num_train // self.batch_size, 1)
            num_iterations = self.num_epochs * iterations_per_epoch

            for batch_index, data in enumerate(self.train_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, metadata, labels, uuid = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                self.loss_history.append(loss.item())
                if batch_index % self.print_every == (self.print_every - 1):  # print every 2000 mini-batches
                    self.print_and_log(f'(Time {(time.time() - start_time):.2f} sec;) [{epoch + 1}/{self.num_epochs}, {batch_index + 1:5d}/{iterations_per_epoch}] loss: {running_loss / self.print_every:.3f}')
                    running_loss = 0.0

            # every epoch, get accuracy stats:
            train_acc = self.get_acc(self.train_dataloader)
            test_acc = self.get_acc(self.test_dataloader)
            self.train_acc_history.append(train_acc)
            self.test_acc_history.append(test_acc)
            self.print_and_log(f'Epoch {epoch + 1} stats: Training accuracy: {train_acc}.     Test accuracy: {test_acc}.')

            # also save state:
            self.save_state(epoch=(epoch + 1))
            # make plots
            self.make_solver_plots()

        self.print_and_log('Finished Training')
        self.save_state(best=True)
        self.make_solver_plots()


    def get_acc(self, dataloader):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.test_dataloader:
                images, metadata, labels, uuid = data
                images, labels = images.to(self.device), labels.to(self.device)

                # calculate outputs by running images through the network
                outputs = self.model(images)
                # got to be within 1pct of real:
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += torch.sum(torch.abs(outputs - labels) < 0.01).cpu()

                # real dumb way of setting max...
                if total >= self.num_val_samples:
                    break
        return 100 * correct // total




def run_center_solver(device, config_dict=None, load_checkpoint=None):
    log_path = get_run_log_dir()

    # write config parameters to log
    config_file_path = os.path.join(log_path, "config.csv")
    with open(config_file_path,'w') as config_file:
        config_file_writer = csv.writer(config_file)
        for key, value in config_dict.items():
            print(f"{key}: {value}")
            config_file_writer.writerow([key, value])

    class_model = get_center_model(device)

    criterion = nn.MSELoss()

    # use checkpoint from previous run?
    if load_checkpoint is not None:
        class_model.load_state_dict(torch.load(load_checkpoint, weights_only=True))
        class_model.eval()
        class_model.to(device)

    # should all params in the model be trainable?  or should the CNN be fixed and just the ones in the fully connected layers?:
    optimizer_ft = optim.Adam(class_model.parameters(),
                              lr=config_dict["LEARNING_RATE"],
                              weight_decay=config_dict["WEIGHT_DECAY"])


    solver = CenterSolver(class_model,
                    criterion=criterion,
                    optimizer=optimizer_ft,
                    config_dict=config_dict,
                    device=device,
                    log_dir=log_path,
                    )
    solver.train()
    print(log_path)
    return solver