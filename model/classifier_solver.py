import csv
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import os
from model.model_utils import get_dataloaders, ClassifierModel, get_classifier_model
import torchvision

from model.solver import get_run_log_dir


class ClassifierSolver(object):
    def __init__(self, model, criterion, optimizer, config_dict, **kwargs):
        self.device = kwargs.pop("device", "cuda")
        self.log_dir = kwargs.pop("log_dir", None)
        self.classes = kwargs.pop("classes", None)

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.config_dict = config_dict
        self.max_data = self.config_dict["MAX_DATA"]
        self.batch_size = self.config_dict["BATCH_SIZE"]

        my_dataloaders = self.config_dict["get_dataloaders"]

        dataloaders, dataset_sizes = my_dataloaders(self.batch_size, self.max_data, config_dict=config_dict)
        self.test_dataloader = dataloaders["val"]
        self.train_dataloader = dataloaders["train"]
        self.dataset_sizes = dataset_sizes
        self.num_epochs = self.config_dict["NUM_EPOCHS"]
        self.log_epochs = self.config_dict.get("LOG_EPOCHS")
        self.num_val_samples = self.config_dict["ACC_SAMPLES"] if self.config_dict["ACC_SAMPLES"] else 100
        self.print_every = self.config_dict["PRINT_EVERY"]

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        self._reset()

    def print_and_log(self, msg):
        print(msg)
        if self.log_dir:
            solver_log = os.path.join(self.log_dir, "solver_log.txt")
            with open(solver_log, "a") as log_file:
                log_file.write(msg)
                log_file.write("\n")

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []
        self.test_offset_history = []


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
        self.save_model()
        self.make_solver_plots()
        self.print_class_scores()

    def save_state(self, epoch=None, best=False):
        weights_path = os.path.join(self.log_dir, 'trained_weights.pth')
        best_path = os.path.join(self.log_dir, 'best_weights.pth')
        if epoch:
            if self.log_epochs:
                weights_path = os.path.join(self.log_dir, 'epochs', f'epoch_{epoch + 1}_weights.pth')
            else:
                # don't save epochs unless
                return

        # Save weights
        torch.save(self.model.state_dict(), weights_path)
        if best:
            torch.save(self.best_params, best_path)

    def save_model(self):
        model_path = os.path.join(self.log_dir, 'model.pkl')
        with open(model_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)


    def get_acc(self, dataloader):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in dataloader:
                images, metadata, labels, uuid = data
                images, labels = images.to(self.device), labels.to(self.device)

                # calculate outputs by running images through the network
                outputs = self.model(images)
                # the class with the highest score is what we choose as prediction
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # real dumb way of setting max...
                if total >= self.num_val_samples:
                    break
        return 100 * correct // total


    def make_solver_plots(self, ):
        plt.plot(self.loss_history, 'o')
        window_width = 20
        cumsum_vec = np.cumsum(np.insert(self.loss_history, 0, 0))
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        plt.plot(list(range(0, len(self.loss_history) - window_width + 1)), ma_vec)
        plt.savefig(f"{self.log_dir}/loss.png")
        plt.clf()

        plt.plot(self.train_acc_history, label="train_acc")
        plt.plot(self.test_acc_history, label="test_acc")
        plt.legend(loc="upper left")
        plt.title(
            f"Training and validation accuracy with lr: {self.config_dict['LEARNING_RATE']}, weight decay: {self.config_dict['WEIGHT_DECAY']}")
        plt.savefig(f"{self.log_dir}/accuracy.png")
        plt.clf()


    def print_class_scores(self):
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in self.classes}
        total_pred = {classname: 0 for classname in self.classes}

        # again no gradients needed
        with torch.no_grad():
            for data in self.test_dataloader:
                images, metadata, labels, uuid = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[self.classes[label]] += 1
                    total_pred[self.classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            if total_pred[classname] > 0:
                accuracy = 100 * float(correct_count) / total_pred[classname]
                self.print_and_log(f'Accuracy for class: {classname:5s} is {accuracy:.1f}%  ({int(correct_count)}/{total_pred[classname]})')


def run_class_solver(device, config_dict=None, load_checkpoint=None, classes=None):

    if classes is None:
        classes = ["good", "crack", "crystal", "twin"]

    log_path = get_run_log_dir()

    # write config parameters to log
    config_file_path = os.path.join(log_path, "config.csv")
    with open(config_file_path,'w') as config_file:
        config_file_writer = csv.writer(config_file)
        for key, value in config_dict.items():
            print(f"{key}: {value}")
            config_file_writer.writerow([key, value])

    class_model = get_classifier_model(device, num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()

    # use checkpoint from previous run?
    if load_checkpoint is not None:
        class_model.load_state_dict(torch.load(load_checkpoint, weights_only=True))
        class_model.eval()
        class_model.to(device)

    # should all params in the model be trainable?  or should the CNN be fixed and just the ones in the fully connected layers?:
    optimizer_ft = optim.Adam(class_model.parameters(),
                              lr=config_dict["LEARNING_RATE"],
                              weight_decay=config_dict["WEIGHT_DECAY"])


    solver = ClassifierSolver(class_model,
                              criterion=criterion,
                              optimizer=optimizer_ft,
                              config_dict=config_dict,
                              device=device,
                              log_dir=log_path,
                              classes=classes,
                              )
    solver.train()
    print(log_path)
    return solver