import argparse
from datasets.cifar import DATASET_GETTERS
import os
import sys
import torch
from experiments.train_original import main as train_original_main, parse_args as parse_args_original
from experiments.train_abc import main as train_abc_main, parse_args as parse_args_abc

def parse_args():
    parser = argparse.ArgumentParser(description='Test dataset getters')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name')
    parser.add_argument('--num-labeled', default=4000, type=int, help='Number of labeled data')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size')
    parser.add_argument('--mu', default=7, type=int, help='Ratio of unlabeled to labeled data')
    parser.add_argument('--epochs', default=100, type=int, help='Total training epochs')
    parser.add_argument('--lr', default=0.03, type=float, help='Learning rate')
    parser.add_argument('--alpha', default=1.0, type=float, help='Weight for balanced loss')
    parser.add_argument('--lambda-u', default=1.0, type=float, help='Weight for unsupervised loss')
    parser.add_argument('--threshold', default=0.95, type=float, help='Confidence threshold')
    parser.add_argument('--device', default='cuda', type=str, help='Device to use')
    parser.add_argument('--num-classes', default=10, type=int, help='Number of classes')
    parser.add_argument('--expand-labels', action='store_true', help='Expand labels to match batch size')  # Add this line
    parser.add_argument('--eval-step', default=1024, type=int, help='Evaluation step')  # Add this line
    args = parser.parse_args()
    return args

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Set the random seed
    set_seed(42)

    # Determine the device to use
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Set the arguments for the FixMatch experiment
    args_fixmatch = [
        '--dataset', 'cifar10',
        '--num-labeled', '4000',
        '--batch-size', '64',
        '--mu', '7',
        '--epochs', '100',  # Adjust as needed
        '--lr', '0.03',
        '--threshold', '0.95',
        '--device', device,
        '--num-classes', '10',
        '--expand-labels',  # Add this line
        '--eval-step', '1024'  # Add this line
    ]

    print("Starting FixMatch Training:")
    args_fixmatch = parse_args_original(args_fixmatch)  # Convert list to Namespace
    train_original_main(args_fixmatch)

    # Set the arguments for the ABC experiment
    args_abc = [
        '--dataset', 'cifar10',
        '--num-labeled', '4000',
        '--batch-size', '64',
        '--mu', '7',
        '--epochs', '100',
        '--lr', '0.03',
        '--alpha', '1.0',
        '--lambda-u', '1.0',
        '--threshold', '0.95',
        '--device', device,
        '--num-classes', '10',
        '--expand-labels',  # Add this line
        '--eval-step', '1024'  # Add this line
    ]

    print("Starting ABC Training:")
    args_abc = parse_args_abc(args_abc)  # Convert list to Namespace
    train_abc_main(args_abc)

if __name__ == '__main__':
    main()