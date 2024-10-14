import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import ABCModel
from datasets.your_dataset import YourDataset  # Replace with your actual dataset
from utils import compute_class_weights, class_balanced_loss
from configs.abc_config import *
from utils.misc import AverageMeter

# Initialize datasets
labeled_dataset = YourDataset(split='train', labeled=True, transform=your_transforms)
unlabeled_dataset = YourDataset(split='train', labeled=False, transform=your_transforms)
val_dataset = YourDataset(split='val', transform=your_transforms)

# Data loaders
labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size * mu, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
