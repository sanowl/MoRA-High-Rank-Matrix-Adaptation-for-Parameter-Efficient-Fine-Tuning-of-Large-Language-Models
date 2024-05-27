import torch
import torch.nn as nn
import math

class MoRA(nn.Module):
    def __init__(self, hidden_size: int, rank: int) -> None:
        super(MoRA, self).__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.matrix = nn.Parameter(torch.zeros(rank, rank))
        self.type = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compressed_x = self.compress(x)
        output = torch.matmul(compressed_x, self.matrix)
        decompressed_output = self.decompress(output)
        return decompressed_output