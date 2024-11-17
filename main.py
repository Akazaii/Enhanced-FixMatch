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
                        choices=['fixmatch', 'abc'],
                        help='Experiment to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set the random seed
    set_seed(args.seed)

    if args.experiment == 'fixmatch':
        from experiments.train_fixmatch import main as fixmatch_main
        fixmatch_main()
    elif args.experiment == 'abc':
        from experiments.train_abc import main as abc_main
        abc_main()

if __name__ == '__main__':
    main()
