import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset

# Set seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Parameters
dataset_path = "dataset/CUB_200_2011/images"
train_test_split_file = "dataset/CUB_200_2011/train_test_split.txt"
results_path = "results/"
if not os.path.exists(results_path):
    os.makedirs(results_path)
class_labels = [i for i in range(200)]
train = []
test = []
images = []
labels = []
gpus = [0,1,2,3,4]
train_losses = []
test_losses = []
accuracy = []
i=0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
saved = 1
batch_size = 128
epoch_count = 50
model_name = "efficientnet_b1"

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

# Split data into train and test using default split
train = []
test = []
with open(train_test_split_file, "r") as file:
    lines = file.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].split()
        if int(lines[i][1]) == 1:
            train.append(i)
        else:
            test.append(i) 
X_train = [images[i] for i in train]
y_train = [labels[i] for i in train]
X_test = [images[i] for i in test]
y_test = [labels[i] for i in test]
print("Train size: ", len(X_train))
print("Test size: ", len(X_test))

# Convert to tensor
X_train, y_train = torch.tensor(np.array(X_train), dtype=torch.float32), torch.tensor(np.array(y_train), dtype=torch.int32)
X_test, y_test = torch.tensor(np.array(X_test), dtype=torch.float32), torch.tensor(np.array(y_test), dtype=torch.int32)

# Create dataloaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=len(gpus))
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False, num_workers=len(gpus))

# Resnet-18 Model
model = models.efficientnet_b1(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
last_filter = model.classifier[1].in_features
model.classifier[1] = nn.Linear(last_filter, 200)
nn.DataParallel(model, device_ids=gpus)
model = model.to(device)
print("Model created")
with open(results_path + f"{model_name}.txt", "w") as file:
    file.write(str(model))
    file.write("\n=============================================\n")
    file.write(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}\n")
    file.write(f"Trainable number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
print("Total number of parameters: ", sum(p.numel() for p in model.parameters()))
print("Trainable number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
best_loss = int(1e9)
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
    train_losses.append(running_loss / len(train_loader))
    # Test model
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy.append(100 * correct / total)
    test_losses.append(running_loss / len(test_loader))
    if running_loss < best_loss:
        best_loss = running_loss
        torch.save(model.state_dict(), results_path + f"{model_name}_{epoch_count}_best_checkpoint.pth")
        print(f"Best model saved at epoch {epoch + 1}")

print("Training complete")
print(f"Final accuracy on the test images: {100 * correct / total:.4f}%")

# Save model
torch.save(model.state_dict(), results_path + f"{model_name}_{epoch_count}_last_checkpoint.pth")

# Plot loss
plt.plot(np.arange(epoch_count), train_losses, label="Train Loss", ls='-', marker='*', c='hotpink', ms=6, mec='g')
plt.plot(np.arange(epoch_count), test_losses, label="Test Loss", ls='-', marker='d', c='teal', ms=6, mec='orchid')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(axis='y', alpha=0.75, ls='--', c='c', lw=0.5)
plt.savefig(results_path + f"{model_name}_{epoch_count}_loss.png")
plt.clf()
plt.plot(np.arange(epoch_count), accuracy, label="Accuracy", ls='-', marker='o', c='b', ms=6, mec='g')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(axis='y', alpha=0.75, ls='--', c='c', lw=0.5)
plt.savefig(results_path + f"{model_name}_{epoch_count}_accuracy.png")