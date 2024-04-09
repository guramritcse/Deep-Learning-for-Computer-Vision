import os
import sys
import argparse
import torch
from utils import set_seed, preprocess, get_data_loader
# from ae import AutoEncoder, train, get_loss_function, get_optimizer, test_psnr
# from vae import VAE, train, get_loss_function, get_optimizer, test_psnr
# from dmcnn import DMCNN, train, get_loss_function, get_optimizer, test_psnr
# from deblur import DeblurModel, train, get_loss_function, get_optimizer, test_psnr
# from gan import Generator, Discriminator, train, get_loss_function, get_optimizer, test_psnr
# from lstm import ConvLSTMDeblur, train, get_loss_function, get_optimizer, test_psnr
from advlstm import AdvLSTMDeblur, train, get_loss_function, get_optimizer, test_psnr

if __name__ == '__main__':

    # Parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action="store_true", default=False)
    parser.add_argument('--checkpoint', action="store", default=None)
    parser.add_argument('--resume', action="store_true", default=False)
    args = vars(parser.parse_args())

    if args["test"]:
        f = open("output_test.txt", "w")
        sys.stdout = f
    else:
        f = open("output_adv_8_4.txt", "w")
        sys.stdout = f

    # File paths and system parameters
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    train_dataset_path = "train/train_sharp"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Testing mode
    if args["test"]:
        print("Testing mode")
        if args["checkpoint"] is None:
            print("Please provide a checkpoint file")
            sys.exit()
        checkpoint = args["checkpoint"]
        print("Checkpoint file: ", checkpoint)
        toks = checkpoint.split("_")
        model_name = toks[0]
        if model_name == "ConvLSTMDeblur":
            model = ConvLSTMDeblur(input_channels=3, hidden_channels=64, kernel_size=3)
        elif model_name == "AdvLSTMDeblur":
            layers = int(toks[-1].split(".")[0])
            model = AdvLSTMDeblur(input_channels=3, hidden_channels=64, num_layers=layers, kernel_size=3)
        model.load_state_dict(torch.load(checkpoint))
        model = model.to(device)
        avg_psnr = test_psnr(model, device, "mp2_test/custom_test/testing/")
        print(f"Average PSNR between corresponding images: {avg_psnr} dB")
        exit()

    # set seed for reproducibility
    set_seed(42)

    # Setup
    batch_size = 8
    num_layers = 4
    num_epochs = 40
    X_train, y_train = preprocess(train_dataset_path, 240, -1, 8, 0, 0)
    train_loader = get_data_loader(X_train, y_train, batch_size)
    print("Images loaded")
    print("Number of training images: ", len(X_train))
    # model = AutoEncoder(in_channels=3, out_channels=3)
    # model = VAE(latent_dim=100)
    # model = DMCNN()
    # model = (Discriminator(3, 256, 448), Generator(100, 3, 256, 448))
    # model = ConvLSTMDeblur(input_channels=3, hidden_channels=64, kernel_size=3)
    model = AdvLSTMDeblur(input_channels=3, hidden_channels=64, num_layers=num_layers, kernel_size=3)

    print("Model: ", model.__class__.__name__)
    print("Batch size: ", batch_size)
    if args["resume"]:
        print("Resuming training")
        model.load_state_dict(torch.load(args["checkpoint"]))
    loss_fn = get_loss_function()
    optimizer = get_optimizer(model)
    print("Total number of parameters: ", sum(p.numel() for p in model.parameters()))
    
    # Train the model
    train(model, train_loader, batch_size, loss_fn, optimizer, num_epochs, device)

    # Test the model
    avg_psnr = test_psnr(model, device, f"mp2_test/custom_test/output_{model.__class__.__name__}_{batch_size}_{model.num_layers}/")
    print(f"Average PSNR between corresponding images: {avg_psnr} dB")
