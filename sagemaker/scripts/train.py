from __future__ import print_function
import argparse
import sys
import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from model import SimpleNet


def _get_data_loader(batch_size, data_dir, filename):
    """Instantiate a PyTorch data loader
    """
    data = pd.read_csv(os.path.join(data_dir, filename), header=None, names=None)

    # Load labels from first column
    labels = torch.from_numpy(data[[0]].values).float().squeeze()
    X = torch.from_numpy(data.drop([0], axis=1).values).float()

    tensor_ds = torch.utils.data.TensorDataset(X, labels)
    
    return torch.utils.data.DataLoader(tensor_ds, batch_size=batch_size)


def _train(model, data_loader, epochs, optimizer, criterion, device):
    """Perform training on provided hyperparameters

    :param model: PyTorch model to train
    :param train_loader: PyTorch DataLoader that should be used during training.
    :param epochs: Total number of epochs to train for.
    :param optimizer: Optimizer to use during training.
    :param criterion: Loss function to optimize. 
    :param device: Where the model and data should be loaded (gpu or cpu).
    """
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(data_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad() # Zero accumulated gradients
            output = model(data) # Forward propagation
            loss = criterion(output, target) # Calculate loss
            loss.backward() # Backward propagation
            optimizer.step()
            total_loss += loss.item()

        # print loss stats
        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(data_loader)))


if __name__ == '__main__':
    # SageMaker parameters are injected via environment variables.
    parser = argparse.ArgumentParser()
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    # Custom parameters that we can pass from notebook instance
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--input_dim', type=int, default=2, metavar='ID', help='input dimension for training data')
    parser.add_argument('--hidden_dim', type=int, default=10, metavar='HD', help='hidden dimension for neural network')
    parser.add_argument('--output_dim', type=int, default=1, metavar='OD', help='output dimension for neural network')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    train_data_loader = _get_data_loader(args.batch_size, args.data_dir, 'train.csv')

    model = SimpleNet(args.input_dim, args.hidden_dim, args.output_dim).to(device)

    # Save model's metadata that is pertaining to its architecture
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_dim': args.input_dim,
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim
        }
        torch.save(model_info, f)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss() # Binary cross entropy loss

    _train(model, train_data_loader, args.epochs, optimizer, criterion, device)

    # Save model's trained parameters
    path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.cpu().state_dict(), path)
