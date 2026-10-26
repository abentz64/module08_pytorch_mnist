# Importing necessary libraries
import torch.nn as nn  # Neural network module
import matplotlib.pyplot as plt  # Plotting library
import argparse
import torch  # PyTorch library
from utilities import * #import all classes/functions from the utilities file
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

'''
Create a concise `run.py` file with proper scripts to train the model by running 
`python run.py` along with the appropriate three requested flags (i.e., arguments) 
in the terminal Command Line Interface (CLI), similar to the practices in the lecture.
 The short flags must be `e,` `b,` and `l` for `num_epochs,` `batch_size,` and 
 `learning_rate,` respectively.
'''


def main():
    parser = argparse.ArgumentParser(description="module 8 assignment")
    parser.add_argument('-e', "--num_epochs", type= int, help = 'Number of epochs', default = 2)
    parser.add_argument("-b", "--batch_size", type= int, help = 'Batch size for images', default = 1000)
    parser.add_argument("-l", "--learning_rate", type= float, help = 'Learning Rate for training', default = .001)
    #parse the arguments
    args = parser.parse_args()
    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    num_epochs = args.num_epochs  # Number of times the entire dataset is passed through the model
    batch_size = args.batch_size  # Number of samples per batch to be passed through the model
    learning_rate = args.learning_rate  # Step size for parameter updates

    #print statements
    print("-"*25)
    print(f"\nTraining for {num_epochs} epochs with batch size {batch_size} and learning rate {learning_rate}\n")
    print("-"*25)

    # MNIST dataset
    # Download & load the training dataset, applying a transformation to convert images to PyTorch tensors
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

    # Download & load the test dataset, applying a transformation to convert images to PyTorch tensors
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    # Data loader
    # DataLoader provides an iterable over the dataset with support for batching, shuffling, & parallel data loading
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    # DataLoader for the test dataset, used for evaluating the model
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Create an instance of the model & move it to the configured device (GPU/CPU)
    model = ConvNet().to(device)

    # Print the model for observation
    #print(model)

    # Loss & optimizer
    # CrossEntropyLoss combines nn.LogSoftmax & nn.NLLLoss in one single class
    criterion = nn.CrossEntropyLoss()
    # Adam optimizer with the specified learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training the model
    model, loss_list, acc_train, acc_test = training(model, num_epochs,
                                                    train_loader, test_loader, criterion, optimizer)
    # Make some plots!
    plot(loss_list, acc_train, acc_test)
    pass


if __name__ == "__main__":
    main()
