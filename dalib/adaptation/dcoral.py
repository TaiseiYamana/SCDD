import torch
import torch.nn as nn
import numpy as np

class DeepCoralLoss(nn.Module):
    def __init__(self):
        super(DeepCoralLoss, self).__init__()

    def compute_convariance(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute Covariance matrix of the input data
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n = logits.size(0)  # batch_size

        id_row = torch.ones(n).resize(1, n).to(device=device)
        sum_column = torch.mm(id_row, logits)
        mean_column = torch.div(sum_column, n)
        term_mul_2 = torch.mm(mean_column.t(), mean_column)
        d_t_d = torch.mm(logits.t(), logits)
        c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

        return c        

    def forward(self, source_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
        d = source_logits.size(1) # num_classes
        
        source_c = self.compute_convariance(source_logits)
        target_c = self.compute_convariance(target_logits)

        dcoral_loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))
        dcoral_loss = dcoral_loss / (4 * d * d)
        return dcoral_loss
