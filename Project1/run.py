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
import torchvision.transforms as transforms
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', action="store_true", default=False)
parser.add_argument('--checkpoint', action="store", default=None)
args = vars(parser.parse_args())

# file paths and common test/train parameters
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
dataset_path = "dataset/CUB_200_2011/images"
train_test_split_file = "dataset/CUB_200_2011/train_test_split.txt"
image_id_file = "dataset/CUB_200_2011/images.txt"
results_path = "results/"
dict_images_id = {}
dict_images_train_test = {}
class_labels = [i for i in range(200)]
X_train = []
X_test = []
y_train = []
y_test = []
i = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 512
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.4)

# Set seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args["test"]:
    if args["checkpoint"] is None:
        print("no checkpoint provided")
        exit(1)
    tokens = args["checkpoint"].split("/")
    tokens = tokens[-1].split("_")
    model_name = '_'.join(tokens[:-4]) if tokens[-2]!="last" else '_'.join(tokens[:-3])
    print("Model:", model_name)
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        last_filter = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(last_filter, len(class_labels))
    elif model_name == "dense_net_121_unfreeze" or model_name == "dense_net_121_unfreeze_augmentation":
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.features.transition3.parameters():
            param.requires_grad = True
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        for param in model.features.norm5.parameters():
            param.requires_grad = True     
        last_filter = model.classifier.in_features
        model.classifier = nn.Linear(last_filter,200)
    elif model_name == "dense_net_121_unfreeze_classifier" or model_name == "dense_net_121_unfreeze_classifier_augmentation":
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        for param in model.features.norm5.parameters():
            param.requires_grad = True
        last_filter = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(last_filter, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, len(class_labels))
        )
    else:
        print("Model not found")
        exit(1)

    model.load_state_dict(torch.load(args["checkpoint"]))
    model.to(device)
    model.eval()
    print("Model loaded")
    print("Total number of parameters: ", sum(p.numel() for p in model.parameters()))
    with open(image_id_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.split()
            dict_images_id[int(line[0])] = line[1]
    
    with open(train_test_split_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.split()
            dict_images_train_test[dict_images_id[int(line[0])]] = int(line[1])
            
    for dir in os.listdir(dataset_path):
        for img in os.listdir(f"{dataset_path}/{dir}"):
            np_img = cv2.imread(f"{dataset_path}/{dir}/{img}")
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
            np_img = cv2.resize(np_img, (224, 224))
            np_img = np.array(np_img)
            np_img = np_img.astype('uint8')
            if dict_images_train_test[f"{dir}/{img}"]==0:
                X_test.append(np_img)
                y_test.append(i)
        i+=1

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    X_test = [transform(Image.fromarray(x)) for x in X_test]
    y_test = y_test

    print("Number of images in the test dataset: ", len(X_test))

    X_test, y_test = torch.tensor(np.array(X_test), dtype=torch.float32), torch.tensor(np.array(y_test), dtype=torch.int32)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

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
    print(f"Accuracy on the test images: {100 * correct / total:.4f}%")
    exit(0)

# Parameters
if not os.path.exists(results_path):
    os.makedirs(results_path)
train_losses = []
test_losses = []
train_accuracy = []
test_accuracy = []
saved = 0
epoch_count = 30
model_name = "dense_net_121_unfreeze_augmentation"
augmentation = 1

# Load images and set labels
if not saved:
    with open(image_id_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.split()
            dict_images_id[int(line[0])] = line[1]
    
    with open(train_test_split_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.split()
            dict_images_train_test[dict_images_id[int(line[0])]] = int(line[1])
            
    for dir in os.listdir(dataset_path):
        for img in os.listdir(f"{dataset_path}/{dir}"):
            np_img = cv2.imread(f"{dataset_path}/{dir}/{img}")
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
            np_img = cv2.resize(np_img, (224, 224))
            np_img = np.array(np_img)
            np_img = np_img.astype('uint8')
            if dict_images_train_test[f"{dir}/{img}"]==1:
                X_train.append(np_img)
                y_train.append(i)
            else:
                X_test.append(np_img)
                y_test.append(i)
        i+=1
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
else:
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")

print("Images loaded")
print("Train size: ", len(X_train))
print("Test size: ", len(X_test))

# Transform data
transform_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_2 = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply transform to data
if augmentation:
    X_train = [transform_1(Image.fromarray(x)) for x in X_train] + [transform_2(Image.fromarray(x)) for x in X_train]
    y_train = [y for y in y_train] + [y for y in y_train]
else:
    X_train = [transform_1(Image.fromarray(x)) for x in X_train]
    y_train = y_train

X_test = [transform_1(Image.fromarray(x)) for x in X_test]
y_test = y_test
print("Data transformed")

# Convert to tensor
X_train, y_train = torch.tensor(np.array(X_train), dtype=torch.float32), torch.tensor(np.array(y_train), dtype=torch.int32)
X_test, y_test = torch.tensor(np.array(X_test), dtype=torch.float32), torch.tensor(np.array(y_test), dtype=torch.int32)

# Create dataloaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=6)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False, num_workers=6)

# Create model
print("Model:", model_name)
if model_name == "resnet18":
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    last_filter = model.fc.in_features
    model.fc = nn.Linear(last_filter, len(class_labels))
elif model_name == "mobilenet_v2":
    model = models.mobilenet_v2(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    last_filter = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(last_filter, len(class_labels))
elif model_name == "efficientnet_b0":
    model = models.efficientnet_b0(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    last_filter = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(last_filter, len(class_labels))
elif model_name == "efficientnet_b1":
    model = models.efficientnet_b1(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    last_filter = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(last_filter,len(class_labels))
elif model_name == "dense_net_121":
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    last_filter = model.classifier.in_features
    model.classifier = nn.Linear(last_filter,len(class_labels))
elif model_name == "dense_net_121_finetune":
    model = models.densenet121(pretrained=True)
    last_filter = model.classifier.in_features
    model.classifier = nn.Linear(last_filter,len(class_labels))
elif model_name == "dense_net_121_classifier":
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    last_filter = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(last_filter, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, len(class_labels))
    )
elif model_name == "dense_net_121_unfreeze_classifier" or model_name == "dense_net_121_unfreeze_classifier_augmentation" :
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features.denseblock4.parameters():
        param.requires_grad = True
    for param in model.features.norm5.parameters():
        param.requires_grad = True
    last_filter = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(last_filter, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, len(class_labels))
    )
elif model_name == "dense_net_121_unfreeze" or model_name == "dense_net_121_unfreeze_augmentation":
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features.transition3.parameters():
        param.requires_grad = True
    for param in model.features.denseblock4.parameters():
        param.requires_grad = True
    for param in model.features.norm5.parameters():
        param.requires_grad = True
    last_filter = model.classifier.in_features
    model.classifier = nn.Linear(last_filter,len(class_labels))

# nn.DataParallel(model, device_ids=gpus)
model = model.to(device)
print("Model created")
with open(results_path + f"{model_name}.txt", "w") as file:
    file.write(str(model))
    file.write("\n=============================================\n")
    file.write(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}\n")
    file.write(f"Trainable number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
print("Total number of parameters: ", sum(p.numel() for p in model.parameters()))
print("Trainable number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = optim.Adam(model.parameters(), lr=0.003, betas=(0.9, 0.999))

with open(results_path + f"{model_name}_{epoch_count}_log.txt", "w") as log_file:
    log_file.write(f"Batch size: {batch_size}\n")
    best_loss = int(1e9)
    best_accuracy = 0
    for epoch in range(epoch_count):
        # Train model
        model.train()
        correct = 0
        total = 0
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
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_accuracy.append(100 * correct / total)
        print(f"[Epoch {epoch + 1}] train loss: {running_loss / len(train_loader):.6f}")
        print(f"[Epoch {epoch + 1}] train accuracy: {100 * correct / total:.4f}%")
        log_file.write(f"[Epoch {epoch + 1}] train loss: {running_loss / len(train_loader):.6f}\n")
        log_file.write(f"[Epoch {epoch + 1}] train accuracy: {100 * correct / total:.4f}%\n")
        train_losses.append(running_loss / len(train_loader))
        # Test model
        model.eval()
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
        test_accuracy.append(100 * correct / total)
        test_losses.append(running_loss / len(test_loader))
        print(f"[Epoch {epoch + 1}] test loss: {running_loss / len(test_loader):.6f}")
        print(f"[Epoch {epoch + 1}] test accuracy: {100 * correct / total:.4f}%")
        log_file.write(f"[Epoch {epoch + 1}] test loss: {running_loss / len(train_loader):.6f}\n")
        log_file.write(f"[Epoch {epoch + 1}] test accuracy: {100 * correct / total:.4f}%\n")
        if running_loss < best_loss:
            best_loss = running_loss
            torch.save(model.state_dict(), results_path + f"{model_name}_{epoch_count}_best_loss_checkpoint.pth")
            print(f"Best loss model saved at epoch {epoch + 1}")
        if best_accuracy < 100 * correct / total:
            best_accuracy = 100 * correct / total
            torch.save(model.state_dict(), results_path + f"{model_name}_{epoch_count}_best_accuracy_checkpoint.pth")
            print(f"Best accuracy model saved at epoch {epoch + 1}")

print("Training complete")
print(f"Final accuracy on the test images: {100 * correct / total:.4f}%")

# Save model
torch.save(model.state_dict(), results_path + f"{model_name}_{epoch_count}_last_checkpoint.pth")

# Plot loss
plt.plot(np.arange(1, epoch_count+1), train_losses, label="Train Loss", ls='-', marker='o', c='hotpink', ms=6, mec='g')
plt.plot(np.arange(1, epoch_count+1), test_losses, label="Test Loss", ls='-', marker='*', c='teal', ms=8, mec='orchid')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(axis='y', alpha=0.75, ls='--', c='c', lw=0.5)
plt.savefig(results_path + f"{model_name}_{epoch_count}_loss.png")
plt.clf()
plt.plot(np.arange(1, epoch_count+1), train_accuracy, label="Train Accuracy", ls='-', marker='o', c='red', ms=6, mec='black')
plt.plot(np.arange(1, epoch_count+1), test_accuracy, label="Test Accuracy", ls='-', marker='*', c='purple', ms=8, mec='orange')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.yticks(range(0, 101, 10))
plt.legend()
plt.grid(axis='y', alpha=0.75, ls='--', c='c', lw=0.5)
plt.savefig(results_path + f"{model_name}_{epoch_count}_accuracy.png")