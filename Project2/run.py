import os
import sys
import argparse
import torch
from utils import set_seed, preprocess, get_data_loader
import lstm 
import advlstm

if __name__ == '__main__':
    # Parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action="store_true", default=False)
    parser.add_argument('--checkpoint', action="store", default=None)
    parser.add_argument('--resume', action="store_true", default=False)
    args = vars(parser.parse_args())

    # Redirect output to a log file if not in test mode
    if not args["test"]:
        f = open("log.txt", "w")
        sys.stdout = f

    # File paths and system parameters
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
        checkpoint_name = checkpoint.split("/")[-1]
        toks = checkpoint_name.split("_")
        model_name = toks[0]
        if model_name == "ConvLSTMDeblur":
            print("Testing ConvLSTMDeblur model")
            model = lstm.ConvLSTMDeblur(input_channels=3, hidden_channels=64, kernel_size=3)
            model.load_state_dict(torch.load(checkpoint))
            model = model.to(device)
            avg_psnr = lstm.test_psnr(model, device, "mp2_test/custom_test/testing/")
        elif model_name == "AdvLSTMDeblur":
            print("Testing AdvLSTMDeblur model")
            layers = int(toks[-1].split(".")[0])
            model = advlstm.AdvLSTMDeblur(input_channels=3, hidden_channels=64, num_layers=layers, kernel_size=3)
            model.load_state_dict(torch.load(checkpoint))
            model = model.to(device)
            avg_psnr = advlstm.test_psnr(model, device, "mp2_test/custom_test/testing/")
        else:
            print("Invalid model name")
            exit()
        print(f"Average PSNR between corresponding images: {avg_psnr} dB")
        exit()

    # Set seed for reproducibility
    set_seed(42)

    # Define model, loss function, optimizer and other hyperparameters
    batch_size = 8
    num_layers = 4
    num_epochs = 20
    X_train, y_train = preprocess(train_dataset_path, 240, -1, 1, 0, 0)
    train_loader = get_data_loader(X_train, y_train, batch_size)
    print("Images loaded")
    print("Number of training images: ", len(X_train))
    # model = lstm.ConvLSTMDeblur(input_channels=3, hidden_channels=64, kernel_size=3)
    model = advlstm.AdvLSTMDeblur(input_channels=3, hidden_channels=64, num_layers=num_layers, kernel_size=3)

    print("Model: ", model.__class__.__name__)
    print("Batch size: ", batch_size)

    # Load checkpoint if resuming training
    if args["resume"]:
        if args["checkpoint"] is None:
            print("Please provide a checkpoint file")
            sys.exit()
        print("Checkpoint file: ", args["checkpoint"])
        print("Resuming training")
        model.load_state_dict(torch.load(args["checkpoint"]))
    print("Total number of parameters: ", sum(p.numel() for p in model.parameters()))
    
    if model.__class__.__name__ == "ConvLSTMDeblur":
        loss_fn = lstm.get_loss_function()
        optimizer = lstm.get_optimizer(model)
        # Train the model
        lstm.train(model, train_loader, batch_size, loss_fn, optimizer, num_epochs, device)
    elif model.__class__.__name__ == "AdvLSTMDeblur":
        loss_fn = advlstm.get_loss_function()
        optimizer = advlstm.get_optimizer(model)
        # Train the model
        advlstm.train(model, train_loader, batch_size, loss_fn, optimizer, num_epochs, device)
    else:
        print("Invalid model name")
        exit()
