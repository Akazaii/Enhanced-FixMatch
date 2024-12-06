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
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--moco-alpha', type=float, default=0.5, help='Weight for MoCo loss in the total loss')
    parser.add_argument('--moco-k-size', type=int, default=65536, help='Queue size for negative keys in MoCo')
    parser.add_argument('--moco-momentum', type=float, default=0.999, help='Momentum for updating key encoder')
    parser.add_argument('--moco-temperature', type=float, default=0.07, help='Temperature for contrastive loss')
    parser.add_argument('--mask-threshold-initial', type=float, default=0.1, help='Initial mask threshold for dynamic masking')
    parser.add_argument('--mask-threshold-max', type=float, default=0.7, help='Max mask threshold for dynamic masking')
    parser.add_argument('--q-aug-type', type=str, default='weak', choices=['weak', 'strong'], help='Augmentation type for query (im_q)')
    parser.add_argument('--k-aug-type', type=str, default='weak', choices=['weak', 'strong'], help='Augmentation type for key (im_k)')

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
        '--out', args.out,
        '--resume',str(args.resume),
        '--moco-alpha', str(args.moco_alpha),
        '--moco-k-size', str(args.moco_k_size),
        '--moco-momentum', str(args.moco_momentum),
        '--moco-temperature', str(args.moco_temperature),
        '--mask-threshold-initial', str(args.mask_threshold_initial),
        '--mask-threshold-max', str(args.mask_threshold),
        '--q-aug-type', args.q_aug_type,
        '--k-aug-type', args.k_aug_type
    ]

    from scripts.train_original import main as train_main
    train_main(train_args)

if __name__ == '__main__':
    main()
