from packaging import version
import torch
from torch import nn
import numpy as np


class PatchNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k, num_patches=512):
        """
        feat_q: Pred features [B, C, H, W]
        feat_k: GT   features [B, C, H, W]
        num_patches: int number of Patches
        """
        batch_size, dim, H, W = feat_q.shape
        if torch.isnan(feat_q).any() or feat_q.max()==0:
            return feat_q.flatten() + 100
        # Sample the Patches:
        # [B, C, H, W] --> [B, C, HW] --> [B, C, P] --> [B, P, C] --> [B*P, C]
        feat_q_flatten = torch.flatten(feat_q, start_dim=2, end_dim=-1)  # [B, C, H, W] --> [B, C, HW]
        feat_k_flatten = torch.flatten(feat_k, start_dim=2, end_dim=-1)  # [B, C, H, W] --> [B, C, HW]
        patch_id = np.random.permutation(H*W)  # random numbers in this range (0->HW) with size [HW]
        patch_id = patch_id[:int(min(num_patches, H*W))]
        patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat_q.device)
        
        # [B, C, HW] --> [B, C, P] --> [B, P, C]
        feat_q_sampled = feat_q_flatten[:, :, patch_id].permute(0, 2, 1)
        feat_k_sampled = feat_k_flatten[:, :, patch_id].permute(0, 2, 1)
        #feat_k_sampled = feat_k_sampled.detach()

        # Calculate the cosine similarity
        feat_k_sampled = feat_k_sampled.transpose(2, 1)  # [B, P, C]-->[B, C, P]
        A_norm = feat_q_sampled / feat_q_sampled.norm(dim=-1, keepdim=True)
        B_norm = feat_k_sampled / feat_k_sampled.norm(dim=1, keepdim=True)
        cosine_similarity = torch.matmul(A_norm, B_norm)  # [B, P, P]
        out = cosine_similarity.flatten(0, 1)

        tgt = torch.zeros(out.size(0), dtype=torch.long, device=feat_q_sampled.device)  # [B*P]
        tgt = torch.arange(start=0, end=min(num_patches, H*W)).repeat(batch_size).long().to(feat_q_sampled.device)
        loss = self.cross_entropy_loss(out, tgt)  # [B*P]
        
        return loss