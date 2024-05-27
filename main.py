import torch
import torch.nn as nn
from typing import Optional, Callable

class MoRA(nn.Module):
    def __init__(self, hidden_size: int, rank: int, group_type: int = 0) -> None:
        super(MoRA, self).__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.matrix = nn.Parameter(torch.zeros(rank, rank))
        self.group_type = group_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compressed_x = self.compress(x)
        output = torch.matmul(compressed_x, self.matrix)
        decompressed_output = self.decompress(output)
        return decompressed_output

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x_padded = torch.cat([x, torch.zeros(batch_size, seq_len, self.hidden_size - x.shape[2], device=x.device)], dim=2)
        
        if self.group_type == 0:
            compressed_x = x_padded.view(batch_size, seq_len, -1, self.rank).sum(dim=2)
        else:
            compressed_x = x_padded.view(batch_size, seq_len, self.rank, -1).sum(dim=3)
        
        return compressed_x

    def decompress(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        if self.group_type == 0:
            decompressed_x = x.repeat_interleave(self.hidden_size // self.rank, dim=2)
        else:
            decompressed_x = x.repeat(1, 1, self.hidden_size // self.rank)
        
        decompressed_x = decompressed_x[:, :, :self.hidden_size]
        
        return decompressed_x

    def change_group_type(self) -> None:
        self.group_type = 1 - self.group_type

def apply_mora(model: nn.Module, hidden_size: int, rank: int, merge_steps: int) -> Callable[[int], None]:
    """
    Apply MoRA to the linear layers of the model.

    Args:
        model (nn.Module): The model to apply MoRA to.
        hidden_size (int): The hidden size of the model.
        rank (int): The rank of the MoRA matrix.
        merge_steps (int): The number of steps between merging and resetting.

    Returns:
        update_model (Callable[[int], None]): A function to update the model at each step.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            mora = MoRA(hidden_size, rank)
            setattr(model, name, nn.Sequential(mora, module))

    def merge_and_reset(model: nn.Module) -> None:
        """
        Merge the MoRA matrix into the linear layer weights and reset the MoRA matrix.
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential) and len(module) == 2 and isinstance(module[0], MoRA):
                mora = module[0]
                linear = module[1]
                weight = linear.weight.data
                
                if mora.group_type == 0:
                    weight_merged = weight.view(mora.hidden_size // mora.rank, mora.rank, -1).transpose(0, 1).reshape(mora.hidden_size, -1)
                else:
                    weight_merged = weight.view(mora.rank, mora.hidden_size // mora.rank, -1).transpose(0, 1).reshape(mora.hidden_size, -1)
                
                linear.weight.data.copy_(weight_merged)
                mora.matrix.data.zero_()
                mora.change_group_type()

    def reset_optimizer(optimizer: torch.optim.Optimizer) -> None:
        """
        Reset the learning rate of the optimizer to the initial value.
        """
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr']

    def update_model(step: int) -> None:
        """
        Update the model at each step by merging and resetting if necessary.
        """
        if step % merge_steps == 0:
            merge_and_reset(model)
            reset_optimizer(optimizer)

    return update_model

class MoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, group_type: int = 0, bias: bool = True) -> None:
        super(MoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.group_type = group_type
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.mora = MoRA(in_features, rank, group_type)

    def reset_parameters(self) -> None:
        """
        Initialize the weight and bias parameters.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the MoRALinear layer.
        """
        x = self.mora(x)
        return nn.functional.linear(x, self.weight, self.bias)

    def change_group_type(self) -> None:
        """
        Change the group type of the MoRA module.
        """
        self.mora.change_group_type()

    def merge_weights(self) -> None:
        """
        Merge the MoRA matrix into the linear layer weights.
        """
        weight = self.weight.data
        if self.mora.group_type == 0:
            weight_merged = weight.view(self.out_features // self.rank, self.rank, -1).transpose(0, 1).reshape(self.out_features, -1)
        else:
            weight_merged = weight.view(self.rank, self.out_features // self.rank, -1).transpose(0, 1).reshape(self.out_features, -1)
        self.weight.data.copy_(weight_merged)
        self.mora.matrix.data.zero_()

def apply_mora_linear(model: nn.Module, rank: int, group_type: int = 0) -> None:
    """
    Replace linear layers with MoRALinear layers in the model.

    Args:
        model (nn.Module): The model to apply MoRALinear to.
        rank (int): The rank of the MoRA matrix.
        group_type (int, optional): The initial group type of the MoRA module. Defaults to 0.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            mora_linear = MoRALinear(module.in_features, module.out_features, rank, group_type, module.bias is not None)
            setattr(model, name, mora_linear)

def merge_and_reset_mora_linear(model: nn.Module) -> None:
    """
    Merge the MoRA matrix into the linear layer weights and reset the MoRA matrix for all MoRALinear layers in the model.
    """
    for module in model.modules():
        if isinstance(module, MoRALinear):
            module.merge_weights()
            module.change_group_type()

def update_model_mora_linear(model: nn.Module, optimizer: torch.optim.Optimizer, merge_steps: int) -> Callable[[int], None]:
    """
    Create a function to update the model with MoRALinear layers at each step.

    Args:
        model (nn.Module): The model with MoRALinear layers.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        merge_steps (int): The number of steps between merging and resetting.

    Returns:
        update_model (Callable[[int], None]): A function to update the model at each step.
    """
    def reset_optimizer() -> None:
        """
        Reset the learning rate of the optimizer to the initial value.
        """
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr']

    def update_model(step: int) -> None:
        """
        Update the model at each step by merging and resetting if necessary.
        """
        if step % merge_steps == 0:
            merge_and_reset_mora_linear(model)
            reset_optimizer()

    return update_model