import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torchvision.models as models

# Initial variables
dataset_path = "dataset/CUB_200_2011/images"
class_labels = [i for i in range(200)]
train = []
test = []
images = []
labels = []
i=0

# Load images and set labels
for dir in os.listdir(dataset_path):
    for img in os.listdir(f"{dataset_path}/{dir}"):
        np_img = cv2.imread(f"{dataset_path}/{dir}/{img}")
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        np_img = cv2.resize(np_img, (224, 224))
        np_img = np.array(np_img)
        np_img = np.transpose(np_img, (2, 0, 1)) 
        np_img = np_img.astype('float32')
        images.append(np_img)
        labels.append(i)
    i+=1

print("Images loaded")

# Split data into train and test using default 75% train and 25% test
X_train, X_test, y_train, y_test = train_test_split(images, labels, random_state=42)
print("Train size: ", len(X_train))
print("Test size: ", len(X_test))

# Convert to tensor
train_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(list(zip(X_test, y_test)), batch_size=128, shuffle=False)

# Model
model = models.resnet50(weights="IMAGENET1K_V1")
for param in model.parameters():
    param.requires_grad = False
last_filter = model.fc.in_features
model.fc = nn.Linear(last_filter, 200)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
for epoch in range(10):
    running_loss = 0.0
    print(f"Current Epoch: {epoch + 1}")
    for inputs, labels in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"[Epoch {epoch + 1}] loss: {running_loss / len(train_loader):.6f}")

print("Training complete")

# Test model
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on the test images: {100 * correct / total:.4f}%")

# Save model
torch.save(model.state_dict(), "model.pth")
