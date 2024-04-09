import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
from skimage.io import imread, imsave
from utils import downscale_image, psnr_between_folders
import matplotlib.pyplot as plt

# ConvLSTM cell
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # A convolutional layer that takes the input and hidden states and returns the concatenated output
        self.conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels,
                              out_channels=4 * self.hidden_channels,
                              kernel_size=self.kernel_size,
                              padding=self.padding)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # Concatenate the input tensor and the hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        # Split the combined convolutional output into 4 parts
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)

        # Apply the activation functions
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Compute the next cell state and hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

# ConvLSTM deblurring model
class ConvLSTMDeblur(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMDeblur, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        # ConvLSTM cell and a convolutional layer for the output
        self.conv_lstm_cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)
        self.conv_out = nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        # Sequence length is 1 since image deblurring is a single frame task
        seq_length = 1 
        h, c = torch.zeros(batch_size, self.hidden_channels, height, width).to(x.device), \
               torch.zeros(batch_size, self.hidden_channels, height, width).to(x.device)

        # Add a dimension for sequence length
        x = x.unsqueeze(1)

        # Iterate over the sequence length and get the last hidden state
        for t in range(seq_length):
            h, c = self.conv_lstm_cell(x[:, t, :, :, :], (h, c))

        # Get the output from the last hidden state and apply the output convolutional layer
        out = self.conv_out(h)
        return out

# Plot the losses and PSNR values
def plot_losses(losses, psnrs, batch_size, model):
    epoch_count = len(losses)
    plt.plot(np.arange(1, epoch_count+1), losses, label="Train Loss", ls='-', marker='o', c='hotpink', ms=6, mec='g')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(axis='y', alpha=0.75, ls='--', c='c', lw=0.5)
    plt.savefig(f"{model.__class__.__name__}_train_loss_{batch_size}.png")
    plt.clf()
    plt.plot(np.arange(1, epoch_count+1), psnrs, label="Test PSNR", ls='-', marker='*', c='purple', ms=8, mec='orange')
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.legend()
    plt.grid(axis='y', alpha=0.75, ls='--', c='c', lw=0.5)
    plt.savefig(f"{model.__class__.__name__}_test_psnr_{batch_size}.png")
    plt.clf()

# Train the model 
def train(model, train_loader, batch_size, loss_fn, optimizer, num_epochs, device):
    model = model.to(device)
    model.train()
    prev_loss = 1
    prev_psnr = 0
    losses = []
    psnrs = []
    for epoch in range(num_epochs):
        for X_batch, y_batch in tqdm(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Save the model with the best loss and PSNR values
        if loss.item() < prev_loss:
            print("Best loss checkpoint saved at epoch ", epoch+1)
            torch.save(model.state_dict(), f"{model.__class__.__name__}_best_loss_checkpoint_{batch_size}.pth")
            prev_loss = loss.item()
        losses.append(loss.item())
        psnr_val = test_psnr(model, device, f"mp2_test/custom_test/output_{model.__class__.__name__}_{batch_size}/")
        os.system(f"rm -r mp2_test/custom_test/output_{model.__class__.__name__}_{batch_size}")
        psnrs.append(psnr_val)
        if psnr_val > prev_psnr:
            print("Best PSNR checkpoint saved at epoch ", epoch+1)
            torch.save(model.state_dict(), f"{model.__class__.__name__}_best_psnr_checkpoint_{batch_size}.pth")
            prev_psnr = psnr_val
        plot_losses(losses, psnrs, batch_size, model)
        print("Last checkpoint saved at epoch ", epoch+1)
        torch.save(model.state_dict(), f"{model.__class__.__name__}_last_checkpoint_{batch_size}.pth")
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        print(f"Epoch {epoch+1}, PSNR: {psnr_val}")
    print("Training complete")

# Get the loss function
def get_loss_function():
    return nn.MSELoss()

# Get the optimizer
def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001)

# Test the model and calculate the PSNR values
def test_psnr(model, device, folder2="mp2_test/custom_test/output/"):
    folder1 = "mp2_test/custom_test/sharp/"
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
    return avg_psnr

if __name__ == '__main__':
    model = ConvLSTMDeblur(input_channels=3, hidden_channels=64, kernel_size=3)
    print(model)
    print("Total number of parameters: ", sum(p.numel() for p in model.parameters()))
