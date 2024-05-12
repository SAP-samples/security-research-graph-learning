import torch 
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse

class GraphStructureLearner(torch.nn.Module):
    def __init__(self, input_channels, num_heads, attention_threshold_epsilon, is_heterogeneous=False, metric_type='weighted_cosine'):
        super().__init__()
        self.num_heads = num_heads
        self.attention_threshold_epsilon = attention_threshold_epsilon
        # weighted cosine
        # (h, 1, dim)
        if metric_type == 'weighted_cosine':
            self.weight_tensor = torch.nn.Parameter(torch.Tensor(num_heads, input_channels)).unsqueeze(1).unsqueeze(0) # batching
            self.weight_tensor = torch.nn.Parameter(torch.nn.init.xavier_uniform_(self.weight_tensor))
            # initialize weights
            
            self.forward = self.forward_weighted_cosine
            
    def forward_weighted_cosine(self, X, has_converged=None):
        X = X.unsqueeze(1) # (8, 1, 1000, 128), (1, 4, 1, 128)
        # hadamard product by broadcasting
        # (h, 1, dim) * (1, n, dim) -> (h, n, dim)
        X = torch.multiply(self.weight_tensor, X)
        Xnorm = F.normalize(X, p=2, dim=-1)
        attention = torch.matmul(Xnorm, Xnorm.transpose(2,3).detach())
        attention = torch.mean(attention, dim=1)
        attention[attention < self.attention_threshold_epsilon] = 0
        if has_converged is not None:
            attention[has_converged,...] = 0
        return attention
  