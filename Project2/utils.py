import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.io import imread
from skimage import transform
from scipy.signal import convolve2d
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Downscale images to 256x448
def downscale_image(image, size=(256, 448)):
    return transform.resize(image, size, anti_aliasing=True, preserve_range=False)


# Create a Gaussian kernel of size x size and standard deviation sigma
def create_gaussian_kernel(size, sigma):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


# Apply a Gaussian filter to an image
def apply_gaussian_filter(image, sigma, kernel_size):
    kernel = create_gaussian_kernel(kernel_size, sigma)
    padded_image = np.pad(image, (((kernel_size-1)//2, (kernel_size-1)//2), ((kernel_size-1)//2, (kernel_size-1)//2), (0, 0)), mode='constant', constant_values=0)
    convolved_image = np.zeros_like(image)
    for i in range(image.shape[2]):
        convolved_image[:, :, i] = convolve2d(padded_image[:, :, i], kernel, mode='valid')
    return convolved_image


# Preprocess images i.e. downscale and apply 3 Gaussian filters
def preprocess(folder, categories, limit=-1, step = 1, saved=0, to_save=0):
    if saved == 1:
        X_train = torch.load("X_train.pt")
        y_train = torch.load("y_train.pt")
        return X_train, y_train
    else:
        X_train = []
        y_train = []
        categories = ["0"*(3-len(str(i))) + str(i) for i in range(categories)]
        for c in categories:
            print("Processing category: ", c)
            subfolder = os.path.join(folder, c)
            filenames = os.listdir(subfolder)
            filenames.sort()
            total = 0
            for i in range(0, len(filenames), step):
                filename = filenames[i]
                print("Processing image: ", filename)
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(subfolder, filename)
                    img = imread(img_path)
                    img = downscale_image(img)
                    X_train.append(apply_gaussian_filter(img, sigma=0.3, kernel_size=3))
                    y_train.append(img) 
                    X_train.append(apply_gaussian_filter(img, sigma=1, kernel_size=7))
                    y_train.append(img) 
                    X_train.append(apply_gaussian_filter(img, sigma=1.6, kernel_size=11))
                    y_train.append(img) 
                    total += 1
                if total == limit:
                    break
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
        y_train = torch.tensor(y_train, dtype=torch.float32).permute(0, 3, 1, 2)
        if to_save == 1:
            torch.save(X_train, "X_train.pt")
            torch.save(y_train, "y_train.pt")
        return X_train, y_train


# Calculate the PSNR between two folders
def psnr_between_folders(folder1, folder2):
    psnr_values = []
    
    filenames = os.listdir(folder1)
    
    for filename in filenames:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path1 = os.path.join(folder1, filename)
            img_path2 = os.path.join(folder2, filename)
            img1 = imread(img_path1)
            img2 = imread(img_path2)
            
            psnr = peak_signal_noise_ratio(img1, img2)
            psnr_values.append(psnr)
    
    avg_psnr = sum(psnr_values) / len(psnr_values)
    print("Number of test images: ", len(psnr_values))
    return avg_psnr


# Get the DataLoader object for given X, y and batch_size
def get_data_loader(X, y, batch_size):
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True, num_workers=6)

