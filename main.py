import argparse
import random
import numpy as np
import torch

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['fixmatch', 'enhanced_fixmatch'],
                        help='Experiment to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use')
    parser.add_argument('--num-labeled', type=int, required=True, help='Number of labeled samples')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size')
    parser.add_argument('--mu', type=int, required=True, help='Coefficient of unlabeled batch size')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--threshold', type=float, required=True, help='Pseudo label threshold')
    parser.add_argument('--device', type=str, required=True, help='Device to use')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    # Set the random seed
    set_seed(args.seed)

    # Convert args to a list of strings to pass to train_original.py
    train_args = [
        '--experiment', args.experiment,
        '--seed', str(args.seed),
        '--dataset', args.dataset,
        '--num-labeled', str(args.num_labeled),
        '--batch-size', str(args.batch_size),
        '--mu', str(args.mu),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--threshold', str(args.threshold),
        '--device', args.device,
        '--out', args.out
    ]

    from scripts.train_original import main as train_main
    train_main(train_args)

if __name__ == '__main__':
    main()
