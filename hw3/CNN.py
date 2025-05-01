import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import csv

class CNN(nn.Module):
    def __init__(self, num_classes=5):
        # (TODO) Design your CNN, it can only be less than 3 convolution layers
        super(CNN, self).__init__()
        # convolution layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # fully connected layers
        # input size 224x224x3
        # data 224x224x3 -> 112x112x32 -> 56x56x64 -> 28x28x128
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.3)



    def forward(self, x):
        # (TODO) Forward the model
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # flatten -> fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def train(model: CNN, train_loader: DataLoader, criterion, optimizer, device)->float:
    # (TODO) Train the model and return the average loss of the data, we suggest use tqdm to know the progress
    model.train()
    total_loss = 0
    total_sample_sz = 0
    for images, labels in tqdm(train_loader, desc="Training、、、"):
        # move data to device
        images, labels = images.to(device), labels.to(device)
        # zero the parameter gradients for each batch
        optimizer.zero_grad()
        # forward data
        outputs = model(images)
        # calculate cross entropy loss between outputs and labels
        loss = criterion(outputs, labels)
        # back propagation
        loss.backward()
        # update weights
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_sample_sz += labels.size(0)
    avg_loss = total_loss / total_sample_sz
    return avg_loss



def validate(model: CNN, val_loader: DataLoader, criterion, device)->Tuple[float, float]:
    # (TODO) Validate the model and return the average loss and accuracy of the data, we suggest use tqdm to know the progress
    # no dropout and no gradient calculation
    model.eval()
    total_loss = 0
    total_samples = 0
    correct = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation、、、"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(labels.size(0)):
                if predicted[i] == labels[i]:
                    correct += 1
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy

def test(model: CNN, test_loader: DataLoader, criterion, device):
    # (TODO) Test the model on testing dataset and write the result to 'CNN.csv'
    model.eval()
    res = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing、、、"):
            images, names = batch
            print(names)
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(names)):
                res.append({'id': names[i], 'prediction': predicted[i].item()})
    # store in CSV
    with open('CNN.csv', mode='w', newline='') as csv_file:
        header = ['id', 'prediction']
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(res)
    print("save to CNN.csv")
    return