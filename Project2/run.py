import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.io import imread, imsave
from skimage import transform
from scipy.signal import convolve2d
import argparse
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from model import AutoEncoder


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def downscale_image(image, size=(256, 448)):
    return transform.resize(image, size, anti_aliasing=True, preserve_range=False)


def create_gaussian_kernel(size, sigma):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def apply_gaussian_filter(image, sigma, kernel_size):
    kernel = create_gaussian_kernel(kernel_size, sigma)
    padded_image = np.pad(image, (((kernel_size-1)//2, (kernel_size-1)//2), ((kernel_size-1)//2, (kernel_size-1)//2), (0, 0)), mode='constant', constant_values=0)
    convolved_image = np.zeros_like(image)
    for i in range(image.shape[2]):
        convolved_image[:, :, i] = convolve2d(padded_image[:, :, i], kernel, mode='valid')
    return convolved_image


def preprocess(folder, categories, limit=-1, step = 1, saved=0, to_save=0):
    if saved == 1:
        X_train = torch.load("X_train.pt")
        y_train = torch.load("y_train.pt")
        X_train 
        return X_train, y_train
    else:
        X_train = []
        y_train = []
        categories = ["0"*(3-len(str(i))) + str(i) for i in range(categories)]
        for c in categories:
            subfolder = os.path.join(folder, c)
            filenames = os.listdir(subfolder)
            filenames.sort()
            total = 0
            for i in range(0, len(filenames), step):
                filename = filenames[i]
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
            #     break
            # break
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
        y_train = torch.tensor(y_train, dtype=torch.float32).permute(0, 3, 1, 2)
        if to_save == 1:
            torch.save(X_train, "X_train.pt")
            torch.save(y_train, "y_train.pt")
        return X_train, y_train


def train(model, train_loader, loss_fn, optimizer, num_epochs, device):
    model = model.to(device)
    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in tqdm(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    print("Training complete")


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


def test_psnr(model, device):
    folder1 = "mp2_test/custom_test/sharp/"
    folder2 = "mp2_test/custom_test/output/"
    folder3 = "mp2_test/custom_test/blur/"

    if os.path.exists(folder2):
        os.system("rm -r " + folder2)
    os.makedirs(folder2)

    filenames = os.listdir(folder3)
    model.eval()
    for filename in filenames:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder3, filename)
            img = imread(img_path)

            img = downscale_image(img)
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

            output = model(img)
            output = output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            output = output * 255
            output = output.astype(np.uint8)

            output_path = os.path.join(folder2, filename)
            imsave(output_path, output)

    avg_psnr = psnr_between_folders(folder1, folder2)
    print(f"Average PSNR between corresponding images: {avg_psnr} dB")


if __name__ == '__main__':
    # Parser for command line arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument('--test', action="store_true", default=False)
    # parser.add_argument('--checkpoint', action="store", default=None)
    args = vars(parser.parse_args())

    # set seed for reproducibility
    set_seed(42)

    # File paths and system parameters
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    train_dataset_path = "train/train_sharp"
    test_dataset_path = "mp2_test/custom_test/blur"
    results_path = "results/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Setup
    batch_size = 16
    num_epochs = 10
    X_train, y_train = preprocess(train_dataset_path, 240, -1, 5, 0, 1)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=6)
    print("Images loaded")
    print("Number of training images: ", len(X_train))
    model = AutoEncoder(in_channels=3, out_channels=3)
    # model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print("Total number of parameters: ", sum(p.numel() for p in model.parameters()))
    
    # Train the model
    train(model, train_loader, loss_fn, optimizer, num_epochs, device)

    # Test the model
    test_psnr(model, device)
