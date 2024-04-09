import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
from skimage.io import imread, imsave
from utils import downscale_image, psnr_between_folders
import matplotlib.pyplot as plt

# Define the ConvLSTM cell
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels,
                              out_channels=4 * self.hidden_channels,
                              kernel_size=self.kernel_size,
                              padding=self.padding)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

# Define the ConvLSTM deblurring model
class AdvLSTMDeblur(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers, kernel_size):
        super(AdvLSTMDeblur, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        # Define multiple ConvLSTM layers
        self.conv_lstm_layers = nn.ModuleList()
        for i in range(self.num_layers):
            input_dim = self.input_channels if i == 0 else self.hidden_channels
            self.conv_lstm_layers.append(ConvLSTMCell(input_dim, self.hidden_channels, self.kernel_size))

        # Output layer
        self.output_layer = nn.Conv2d(self.hidden_channels, self.input_channels, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()  # Updated line
        seq_length = 1  # Assume single-frame input

        # Add a dummy dimension for sequence length
        x = x.unsqueeze(1)  # Add dimension at index 1

        h_states = [torch.zeros(batch_size, self.hidden_channels, height, width).to(x.device) for _ in range(self.num_layers)]
        c_states = [torch.zeros(batch_size, self.hidden_channels, height, width).to(x.device) for _ in range(self.num_layers)]

        for t in range(seq_length):
            input_data = x[:, t, :, :, :]
            for layer_idx in range(self.num_layers):
                h_states[layer_idx], c_states[layer_idx] = self.conv_lstm_layers[layer_idx](input_data, (h_states[layer_idx], c_states[layer_idx]))
                input_data = h_states[layer_idx]

        out = self.output_layer(h_states[-1])  # Take the output from the last layer
        return out


def plot_losses(losses, psnrs, batch_size, model):
    epoch_count = len(losses)
    plt.plot(np.arange(1, epoch_count+1), losses, label="Train Loss", ls='-', marker='o', c='hotpink', ms=6, mec='g')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(axis='y', alpha=0.75, ls='--', c='c', lw=0.5)
    plt.savefig(f"{model.__class__.__name__}_train_loss_{batch_size}_{model.num_layers}.png")
    plt.clf()
    plt.plot(np.arange(1, epoch_count+1), psnrs, label="Test PSNR", ls='-', marker='*', c='purple', ms=8, mec='orange')
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.legend()
    plt.grid(axis='y', alpha=0.75, ls='--', c='c', lw=0.5)
    plt.savefig(f"{model.__class__.__name__}_test_psnr_{batch_size}_{model.num_layers}.png")
    plt.clf()
    
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
        if loss.item() < prev_loss:
            print("Best loss checkpoint saved at epoch ", epoch+1)
            torch.save(model.state_dict(), f"{model.__class__.__name__}_best_loss_checkpoint_{batch_size}_{model.num_layers}.pth")
            prev_loss = loss.item()
        losses.append(loss.item())
        psnr_val = test_psnr(model, device, f"mp2_test/custom_test/output_{model.__class__.__name__}_{batch_size}_{model.num_layers}/")
        os.system(f"rm -r mp2_test/custom_test/output_{model.__class__.__name__}_{batch_size}_{model.num_layers}")
        psnrs.append(psnr_val)
        if psnr_val > prev_psnr:
            print("Best PSNR checkpoint saved at epoch ", epoch+1)
            torch.save(model.state_dict(), f"{model.__class__.__name__}_best_psnr_checkpoint_{batch_size}_{model.num_layers}.pth")
            prev_psnr = psnr_val
        plot_losses(losses, psnrs, batch_size, model)
        print("Last checkpoint saved at epoch ", epoch+1)
        torch.save(model.state_dict(), f"{model.__class__.__name__}_last_checkpoint_{batch_size}_{model.num_layers}.pth")
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        print(f"Epoch {epoch+1}, PSNR: {psnr_val}")
    print("Training complete")

def get_loss_function():
    return nn.MSELoss()

def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001)

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
    model = AdvLSTMDeblur(input_channels=3, hidden_channels=64, num_layers=3, kernel_size=3)
    print(model)
    print("Total number of parameters: ", sum(p.numel() for p in model.parameters()))
