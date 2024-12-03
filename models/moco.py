import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wideresnet import WideResNet

class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mask_threshold=0.95, encoder_args=None):
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.mask_threshold = mask_threshold

        if encoder_args is None:
            encoder_args = {}
        self.encoder_q = base_encoder(num_classes=dim, **encoder_args)
        self.encoder_k = base_encoder(num_classes=dim, **encoder_args)

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

    def forward(self, im_q, im_k):
        q = self.encoder_q.features(im_q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k.features(im_k)
            k = nn.functional.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        self._dequeue_and_enqueue(k)

        # Apply masking
        mask = torch.max(torch.softmax(logits, dim=1), dim=1)[0] >= self.mask_threshold
        logits = logits[mask]
        labels = labels[mask]

        if logits.numel() == 0:
            # If logits are empty, return None to indicate no loss should be computed
            return None, None

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