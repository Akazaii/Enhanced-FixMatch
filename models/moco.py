import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wideresnet import WideResNet

class MoCo(nn.Module):
    def __init__(self, base_encoder, num_classes, dim=128, K=65536, m=0.999, T=0.07, alpha=0.5, mask_threshold_initial=0.1, mask_threshold_max=0.7, encoder_args=None):
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.initial_mask_threshold = mask_threshold_initial
        self.max_mask_threshold = mask_threshold_max

        if encoder_args is None:
            encoder_args = {}
        self.encoder_q = base_encoder(num_classes=num_classes, **encoder_args)
        self.encoder_k = base_encoder(num_classes=num_classes, **encoder_args)

        # Freeze the key encoder's parameters initially
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        feature_dim = self.encoder_q.num_features

        # Create projection heads
        self.fc_q = nn.Linear(feature_dim, dim)
        self.fc_k = nn.Linear(feature_dim, dim)

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # Replace the assertion with modulo operation
        if ptr + batch_size > self.K:
            overflow = ptr + batch_size - self.K
            self.queue[:, ptr:] = keys[:batch_size - overflow].T
            self.queue[:, :overflow] = keys[batch_size - overflow:].T
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k, epoch=None, total_epochs=None):
        q_features = self.encoder_q.features(im_q)
        q = self.fc_q(q_features)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k_features = self.encoder_k.features(im_k)
            k = self.fc_k(k_features) 
            k = nn.functional.normalize(k, dim=1)
    # Adjust mask_threshold dynamically
        if epoch is not None and total_epochs is not None:
            self.mask_threshold = self.initial_mask_threshold + (
                (self.max_mask_threshold - self.initial_mask_threshold) * (epoch / total_epochs)
            )
        # Compute cosine similarity between q and k
        cos_sim = torch.einsum('nc,nc->n', [q, k])  # Shape: [N]
        # print(f"Cosine similarities: {cos_sim}")
        # Apply masking based on cosine similarity
        mask = cos_sim >= self.mask_threshold

        # Select samples where mask is True
        q = q[mask]
        k = k[mask]
        if q.numel() == 0:
            return None, None

        # Recompute l_pos and l_neg with masked q and k
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        self._dequeue_and_enqueue(k)

        return logits, labels

    def compute_loss(self, logits, labels):
        # Compute the cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensors_gather = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output
    else:
        return tensor