import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam


class Model():
    def __init__(self):
        self.gpu = torch.device('cuda')

        self.net = Net()

        self.net = torch.nn.DataParallel(self.net)
        self.net.to(self.gpu)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.net.parameters(), lr=0.001)

    def data_to_gpu(self, data):
        image, labels = data

        image = image.to(self.gpu)
        labels = torch.LongTensor(labels).to(self.gpu)

        return image, labels

    def train(self, data):
        image, labels = self.data_to_gpu(data)

        self.optimizer.zero_grad()
        output = self.sigmoid(self.net(image))

        loss = self.loss(output, labels)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def valid(self, data):
        image, labels = self.data_to_gpu(data)

        output = self.sigmoid(self.net(image))

        loss = self.loss(output, labels)

        predictions = np.argmax(output.detach().cpu().numpy(), axis=1)
        targets = labels.detach().cpu().numpy()
        accuracy = 1 - sum(np.logical_xor(predictions, targets)) / len(targets)

        return loss.item(), accuracy


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(in_features=1600, out_features=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        return x
