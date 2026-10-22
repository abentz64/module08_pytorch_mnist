# Importing necessary libraries
import torch  # PyTorch library
import torch.nn as nn  # Neural network module
import matplotlib.pyplot as plt  # Plotting library
import argparse
from utilities import * #import all classes/functions from the utilities file
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
    parser.add_argument("--batch_size", '-b', type= int, help = 'Batch size for images', default = 25)
    parser.add_argument("--learning_rate", '-l', type= float, help = 'Learning Rate for training', default = .0002)
    #parse the arguments
    args = parser.parse_args()
    lr = args.learning_rate
    epochs = args.num_epochs
    batch = args.batch_size
    

    pass 


if __name__ == "__main__":
    main()
