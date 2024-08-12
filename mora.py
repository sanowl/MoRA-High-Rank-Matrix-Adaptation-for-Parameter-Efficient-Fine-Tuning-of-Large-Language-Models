import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, TensorDataset

class MoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.merge_weights = merge_weights

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = self.lora_alpha / self.r

        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        # Adaptive rank
        self.adaptive_rank = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            effective_rank = torch.clamp(self.adaptive_rank, min=1, max=self.r)
            lora_A = self.lora_A[:int(effective_rank), :]
            lora_B = self.lora_B[:, :int(effective_rank)]
        else:
            lora_A = self.lora_A
            lora_B = self.lora_B
        
        result = (self.lora_dropout(x) @ lora_A.t() @ lora_B.t()) * self.scaling
        return result

class MoRALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.mora = MoRALayer(
            in_features,
            out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias) + self.mora(x)

class MoRAModel(nn.Module):
    def __init__(self, base_model: nn.Module, mora_config: Dict[str, Union[int, float, bool]]):
        super().__init__()
        self.base_model = base_model
        self.mora_config = mora_config
        self._replace_linear_layers()

    def _replace_linear_layers(self):
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = self.base_model if parent_name == '' else getattr(self.base_model, parent_name)
                setattr(
                    parent,
                    child_name,
                    MoRALinear(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        **self.mora_config
                    )
                )

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

class MoRAOptimizer(Optimizer):
    def __init__(
        self,
        params: List[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class MoRAScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-5,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [max(self.min_lr, base_lr * (1 - progress)) for base_lr in self.base_lrs]

def apply_mora(
    model: nn.Module,
    r: int = 8,
    lora_alpha: float = 1.0,
    lora_dropout: float = 0.0,
    merge_weights: bool = True,
) -> MoRAModel:
    mora_config = {
        'r': r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'merge_weights': merge_weights,
    }
    return MoRAModel(model, mora_config)

# Example usage
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a base model (e.g., a transformer)
    base_model = nn.TransformerEncoderLayer(d_model=512, nhead=8).to(device)
    
    # Apply MoRA to the base model
    mora_model = apply_mora(base_model, r=16, lora_alpha=32, lora_dropout=0.1).to(device)
    
    # Create optimizer and scheduler
    optimizer = MoRAOptimizer(mora_model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = MoRAScheduler(optimizer, warmup_steps=1000, total_steps=10000, min_lr=1e-5)
    
    # Define loss function
    criterion = nn.MSELoss()

    # Create dummy dataset and dataloader
    input_data = torch.randn(1000, 10, 512).to(device)
    target_data = torch.randn(1000, 10, 512).to(device)
    dataset = TensorDataset(input_data, target_data)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        mora_model.train()
        total_loss = 0
        for batch in data_loader:
            inputs, targets = batch
            outputs = mora_model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    print("Training complete!")

if __name__ == "__main__":
    main()