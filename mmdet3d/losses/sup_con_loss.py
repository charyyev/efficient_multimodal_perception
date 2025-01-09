import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    Adapted from https://github.com/HobbitLong/SupContrast/blob/master/losses.py.
    It also supports the unsupervised contrastive loss in SimCLR
    
    Args:
        temperature (float): Temperature parameter
        base_temperature (float): Base temperature 
    """
    def __init__(self, temperature=0.07, base_temperature = 0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """Compute loss for model. 

        Args:
            features (torch.tensor): features of points.
            labels (torch.tensor): cluster assignment for each point.
            
        Returns:
            A loss scalar.
        """

        N = features.shape[0]
        features = F.normalize(features, dim=-1, p=2)
        
        labels = labels.view(-1, 1)
        if labels.shape[0] != N:
            raise ValueError('Num of labels does not match num of features')
        

        # choose anchor points
        anchor_indices = []
        unique_values = torch.unique(labels)
        for i in unique_values:
            indices = torch.where(labels.squeeze() == i)[0]
            if len(indices) < 10:
                continue
            idx = torch.randint(indices.shape[0], (1, 1)).item()
            anchor_index = indices[idx].item()
            anchor_indices.append(anchor_index)
        
        if len(anchor_indices) == 0:
            return None
        
        anchor_feature = features[anchor_indices]
        if len(anchor_feature.shape) == 1:
            anchor_feature = anchor_feature[None, ...]

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = torch.eq(labels[anchor_indices], labels.T).float()
        
        # mask-out self-contrast cases
        logits_mask = torch.ones_like(mask)
        logits_mask[torch.arange(len(anchor_indices)), anchor_indices] = 0

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs


        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss