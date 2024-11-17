import torch
from collections import Counter
import torch.nn.functional as F

def compute_class_weights(labels, num_classes):
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    class_weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        if count > 0:
            weight = total_samples / (num_classes * count)
        else:
            weight = 0.0  # Handle classes with zero instances
        class_weights.append(weight)
    return torch.FloatTensor(class_weights)

def class_balanced_loss(logits, targets, class_weights):
    """
    Computes the class-balanced cross-entropy loss.
    """
    return F.cross_entropy(logits, targets, weight=class_weights.to(logits.device))

# Masking strategy ABC pseudo label ground truth label multiply mask of 


