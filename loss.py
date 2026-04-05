import torch.nn.functional as F
import torch

def info_nce_loss(z1, z2, temperature=0.2):
    """
    z1, z2: Tensor of shape (N, D)
    Return: scalar InfoNCE loss
    """
    z1 = F.normalize(z1, dim=1)  # shape: [N, D]
    z2 = F.normalize(z2, dim=1)  # shape: [N, D]
    N = z1.size(0)

    representations = torch.cat([z1, z2], dim=0)  # [2N, D]
    similarity_matrix = torch.mm(representations, representations.T)  # [2N, 2N]

    # Create positive pair indices
    labels = torch.arange(N, device=z1.device)
    labels = torch.cat([labels + N, labels])  # [2N]

    # Scale similarity
    similarity_matrix = similarity_matrix / temperature

    # Remove diagonal (self similarity)
    mask = torch.eye(2*N, dtype=torch.bool, device=z1.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

    # Apply cross entropy
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss
