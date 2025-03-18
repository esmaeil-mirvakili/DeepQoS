import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import IOBinClassificationDataSet


class DNN(nn.Module):
    def __init__(self, layers_config):
        super(DNN, self).__init__()
        layers = []
        for in_size, out_size, activation_class in layers_config:
            layers.append(nn.Linear(in_size, out_size))
            if activation_class:
                layers.append(activation_class())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ModelA(DNN):
    def __init__(self, input_size, output_size):
        layers = [
            (input_size, 128, nn.ReLU),
            (128, output_size, None)
        ]
        super(ModelA, self).__init__(layers)


class ModelB(DNN):
    def __init__(self, input_size, output_size):
        layers = [
            (input_size, 256, nn.ReLU),
            (256, output_size, None)
        ]
        super(ModelB, self).__init__(layers)


class ModelC(DNN):
    def __init__(self, input_size, output_size):
        layers = [
            (input_size, 256, nn.ReLU),
            (256, 256, nn.ReLU),
            (256, output_size, None)
        ]
        super(ModelC, self).__init__(layers)


class ModelD(DNN):
    def __init__(self, input_size, output_size):
        layers = [
            (input_size, 256, nn.ReLU),
            (256, 512, nn.ReLU),
            (512, 256, nn.ReLU),
            (256, output_size, None)
        ]
        super(ModelD, self).__init__(layers)


class IONETDenseDNN:
    def __init__(self, path, model_class: DNN = ModelA, lr=0.001, batch_size=16, shuffle=False, output=sys.stdout,
                 threshold=2_000_000,
                 seed=42):
        self.path = path
        self.seed = seed
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.output = output
        self.model = None
        self.train_dataset = IOBinClassificationDataSet(self.path, train_size=0.7, stage='train', threshold=threshold)
        self.val_dataset = IOBinClassificationDataSet(self.path, train_size=0.7, val_size=0.15, stage='val',
                                                      threshold=threshold)
        self.test_dataset = IOBinClassificationDataSet(self.path, train_size=0.7, val_size=0.15, stage='test',
                                                       threshold=threshold)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_class(input_size=self.train_dataset.input_size(), output_size=2).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, epochs=100):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        for epoch in range(epochs):
            self.model.train()  # Set model to training mode
            train_loss, correct, total = 0, 0, 0

            for inputs, labels in train_loader:
                inputs = torch.tensor(inputs, dtype=torch.float32)  # Convert input to tensor
                labels = torch.tensor(labels, dtype=torch.long)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track accuracy
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = 100 * correct / total

            # Validation step
            val_loss, val_acc = self.evaluate_model(val_loader)

            self.output.write(f"Epoch [{epoch + 1}/{epochs}] - "
                              f"Train Loss: {train_loss / len(train_loader):.4f} | Train Acc: {train_acc:.2f}% - "
                              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%\n")
        self.output.write('Test Step:')
        # Test step
        test_loss, test_acc = self.evaluate_model(test_loader)
        self.output.write(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    def evaluate_model(self, dataloader):
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():  # Disable gradient calculations
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
