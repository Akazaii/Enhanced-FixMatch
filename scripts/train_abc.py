# experiments/train_abc.py

import argparse
import os
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from models.abc_model import ABCModel
from dataset.cifar import DATASET_GETTERS  # Adjust import if necessary
from utils import compute_class_weights, class_balanced_loss
from utils.misc import AverageMeter

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

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training with ABC and Masking Strategy')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name')
    parser.add_argument('--num-labeled', default=4000, type=int, help='Number of labeled data')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size')
    parser.add_argument('--mu', default=7, type=int, help='Ratio of unlabeled to labeled data')
    parser.add_argument('--epochs', default=100, type=int, help='Total training epochs')
    parser.add_argument('--lr', default=0.03, type=float, help='Learning rate')
    parser.add_argument('--alpha', default=1.0, type=float, help='Weight for balanced loss')
    parser.add_argument('--lambda-u', default=1.0, type=float, help='Weight for unsupervised loss')
    parser.add_argument('--threshold', default=0.95, type=float, help='Confidence threshold')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', default='cuda', type=str, help='Device to use')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    set_seed(args.seed)
    best_acc = 0

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Prepare datasets and data loaders using DATASET_GETTERS
    num_classes = 10 if args.dataset == 'cifar10' else 100
    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data')

    # Create data loaders
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True)

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size * args.mu,
        shuffle=True,
        num_workers=4,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4)

    # Initialize model
    model = ABCModel(num_classes=num_classes).to(device)

    # Initialize optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Compute initial class weights from the entire labeled dataset
    labeled_labels = [label for _, label in labeled_dataset]
    class_weights = compute_class_weights(labeled_labels, num_classes).to(device)

    # Initialize logging
    writer = SummaryWriter(logdir='logs')

    # Initialize class counts for masking strategy
    class_counts = torch.zeros(num_classes, device=device)

    for epoch in range(args.epochs):
        train_loss = train(args, labeled_loader, unlabeled_loader, model, optimizer,
                           class_weights, class_counts, device)
        test_acc = validate(args, test_loader, model, device)

        scheduler.step()

        # Log to TensorBoard
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('test/acc', test_acc, epoch)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join('checkpoints', f'abc_best_model.pth')
            torch.save(model.state_dict(), save_path)

        print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.2f}%')

    writer.close()

def train(args, labeled_loader, unlabeled_loader, model, optimizer, class_weights, class_counts, device):
    model.train()
    losses = AverageMeter()
    data_loader = zip(labeled_loader, unlabeled_loader)
    for batch_idx, ((l_input, l_target), (u_input_weak, u_input_strong)) in enumerate(data_loader):
        batch_size = l_input.size(0)
        l_input, l_target = l_input.to(device), l_target.to(device)
        u_input_weak, u_input_strong = u_input_weak.to(device), u_input_strong.to(device)

        # Update class counts with labels from the current batch
        for label in l_target:
            class_counts[label] += 1

        # Forward pass for labeled data
        l_logits, l_balanced_logits = model(l_input)
        loss_supervised = F.cross_entropy(l_logits, l_target, reduction='mean')

        # Masking strategy for the balanced loss
        # Identify minority classes based on current class counts
        total_class_counts = class_counts.sum()
        class_frequencies = class_counts / total_class_counts
        # Define a threshold to identify minority classes (e.g., classes with frequency less than mean frequency)
        mean_frequency = class_frequencies.mean()
        minority_classes = class_frequencies < mean_frequency
        minority_class_indices = torch.nonzero(minority_classes).squeeze()

        # Create mask for minority class samples in the batch
        mask_minority = torch.zeros_like(l_target, dtype=torch.bool)
        for cls in minority_class_indices:
            mask_minority |= (l_target == cls.item())

        # Compute balanced loss only for minority class samples
        if mask_minority.sum() > 0:
            loss_balanced = class_balanced_loss(
                l_balanced_logits[mask_minority],
                l_target[mask_minority],
                class_weights)
        else:
            loss_balanced = torch.tensor(0.0, device=device)

        # Generate pseudo-labels
        with torch.no_grad():
            u_logits_weak, _ = model(u_input_weak)
            pseudo_labels = torch.softmax(u_logits_weak, dim=-1)
            max_probs, targets_u = torch.max(pseudo_labels, dim=-1)
            mask = max_probs.ge(args.threshold).float()

        # Forward pass for unlabeled data with strong augmentation
        u_logits_strong, _ = model(u_input_strong)
        loss_unsupervised = (F.cross_entropy(u_logits_strong, targets_u, reduction='none') * mask).mean()

        # Total loss
        total_loss = loss_supervised + args.lambda_u * loss_unsupervised
        if loss_balanced > 0:
            total_loss += args.alpha * loss_balanced

        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses.update(total_loss.item(), batch_size)

    return losses.avg

def validate(args, test_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    accuracy = 100 * correct / total
    return accuracy

if __name__ == '__main__':
    main()
