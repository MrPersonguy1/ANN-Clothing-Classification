# -*- coding: utf-8 -*-
"""
### Imports
"""

import torch
import torch.nn as nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset, Dataset

from torchsummary import summary
from tqdm import tqdm

import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np

print(torch.__version__)

"""
###⌛Dataset Downloading
"""

train_data = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download = True,
    transform = ToTensor(),
    target_transform = None,
)

test_data = datasets.FashionMNIST(
    root = 'data',
    train = False,
    download = True,
    transform = ToTensor(),
    target_transform = None,
)

"""
### Visualization
"""

image_num = 5678 #@param {type:"raw"}
plt.imshow(train_data[image_num][0].squeeze(0), cmap='gray')
plt.title(f'Class: {train_data[image_num][1]}; {train_data.classes[train_data[image_num][1]]}')
plt.show()

"""
### Dataloaders
"""

train_dataloader = DataLoader(train_data, batch_size = 32, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 32, shuffle = False)

images, labels = next(iter(train_dataloader))
print(f'Shape of the images: {images.shape}')
print(f'Shape of the labels: {labels.shape}')
print(f'Number of batches is: {len(iter(train_dataloader))}')

"""
## 🟩Model Building
"""

class MultiLayerPerceptron(nn.Module):
  def __init__(self, input_features=784, output=10):
    super().__init__()
    self.layer_1 = nn.Linear(in_features = input_features, out_features = 256)
    self.layer_2 = nn.Linear(in_features = 256, out_features = 128)
    self.layer_3 = nn.Linear(in_features = 128, out_features = 32)
    self.layer_4 = nn.Linear(in_features = 32, out_features = output)

  def forward(self, x):
    x = self.relu(self.layer_1(x))
    x = self.relu(self.layer_2(x))
    x = self.relu(self.layer_3(x))
    x = self.output(x)
    return x

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_features=784, output=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, 256)
            nn.Linear(256, 128)
            nn.Linear(128, 32)
            nn.Linear(32, output)
        )

    def forward(self, x):
        return self.layers(x)


model = MultiLayerPerceptron()

"""
### Model Parameters
"""

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.001)

"""
## 🟥Training
"""

def train_loop(train_dataloader, model, loss_fn, optimizer, epochs):
  model.train()
  train_loss = []

  for epoch in range(epochs):
    train_loss_epoch = 0
    for image, label in tqdm(train_dataloader, desc="Training Model"):
      optimizer.zero_grad()

      pred = model(image)
      loss = loss.fn(pred, label)
      loss.backward()
      train_loss_epoch += loss.item()
      optimizer.step()

    avg_loss = train_loss_epoch / len(train_dataloader)
    train_loss.append(avg_loss)
      
  return train_loss

losses = train_loop(train_dataloader, model, loss_fn, optimizer, epochs=10)

epoch_list = list(range(1, 11))
plt.plot(epoch_list, losses)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

"""
## 🟪Testing
"""

def accuracy(correct, total):
  return correct/total * 100

def test_loop(test_dataloader, model):
  model.eval()
  correct = 0
  total = 0

  with torch.no_grad():
    for image, label in tqdm(test_dataloader, desc="Testing Model"):
      pred = model(image)

      correct += (pred.argmax(1) == label).type(torch.float).sum().item()
      total += len(label)

    print(f'Accuracy: {accuracy(correct, total)}')

test_loop(test_dataloader, model)

rand_idx = torch.randint(0, len(test_data), (1,)).item()
image, label = test_data[rand_idx]

prediction = model(image)

pred_idx = prediction[0].argmax().item()

plt.figure(figsize=(5,5))
plt.title(f'Prediction: {test_data.classes[pred_idx]} | Correct Label: {test_data.classes[label]}')
plt.imshow(image[0].squeeze(), cmap='gray')
plt.show()
