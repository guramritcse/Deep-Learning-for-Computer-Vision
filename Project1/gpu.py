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
from torch.utils.data import DataLoader, TensorDataset

# Initial variables
dataset_path = "dataset/CUB_200_2011/images"
class_labels = [i for i in range(200)]
train = []
test = []
images = []
labels = []
gpus = [1,2,3,4]
i=0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
saved = 1

# Load images and set labels
if not saved:
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
    np.save("images.npy", images)
    np.save("labels.npy", labels)
else:
    images = np.load("images.npy")
    labels = np.load("labels.npy")


print("Images loaded")

# Split data into train and test using default 75% train and 25% test
X_train, X_test, y_train, y_test = train_test_split(images, labels, random_state=42)
print("Train size: ", len(X_train))
print("Test size: ", len(X_test))

# Convert to tensor
X_train, y_train = torch.tensor(np.array(X_train), dtype=torch.float32), torch.tensor(np.array(y_train), dtype=torch.int32)
X_test, y_test = torch.tensor(np.array(X_test), dtype=torch.float32), torch.tensor(np.array(y_test), dtype=torch.int32)

# Create dataloaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True, num_workers=len(gpus))
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False, num_workers=len(gpus))

# Model
model = models.resnet152(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
last_filter = model.fc.in_features
model.fc = nn.Linear(last_filter, 200)
nn.DataParallel(model, device_ids=gpus)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
epoch_count = 20
for epoch in range(epoch_count):
    running_loss = 0.0
    print(f"Current Epoch: {epoch + 1}")
    for inputs, labels in tqdm(train_loader):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
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
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on the test images: {100 * correct / total:.4f}%")

# Save model
torch.save(model.state_dict(), "model.pth")
