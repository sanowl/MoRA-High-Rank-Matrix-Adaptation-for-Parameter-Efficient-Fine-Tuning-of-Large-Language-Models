import math
import logging
from typing import Dict, List, Optional, Tuple, Union
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from transformers import PreTrainedModel, PreTrainedTokenizer, BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.validate_config()

    def __getattr__(self, name):
        return self.config.get(name, None)

    def validate_config(self):
        required_params = ['lr', 'num_epochs', 'base_model_name', 'num_experts', 'expert_hidden_size']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"{param} is required in the configuration file")
        logger.info(f"Loaded configuration: {self.config}")

    def get_optimizer_params(self):
        return self.config.get('optimizer_params', {})

class DynamicExpert(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        return self.fc2(x)

class SparseMoE(nn.Module):
    def __init__(self, num_experts, input_size, hidden_size, output_size, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([DynamicExpert(input_size, hidden_size, output_size) for _ in range(num_experts)])
        self.gate = nn.Linear(input_size, num_experts)

    def forward(self, x):
        gate_logits = self.gate(x)
        weights, indices = torch.topk(gate_logits, self.k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        expert_outputs = []
        for i in range(self.k):
            expert_output = torch.stack([self.experts[j](x[i]) for i, j in enumerate(indices[:, i])])
            expert_outputs.append(expert_output)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)
        output = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=1)
        return output

class AdaptiveComputationTime(nn.Module):
    def __init__(self, input_size, hidden_size, max_steps):
        super().__init__()
        self.max_steps = max_steps
        self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.halting = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h, c = torch.zeros(batch_size, self.rnn.hidden_size).to(x.device), torch.zeros(batch_size, self.rnn.hidden_size).to(x.device)
        halting_probability = torch.zeros(batch_size, 1).to(x.device)
        remainders = torch.zeros(batch_size, 1).to(x.device)
        n_updates = torch.zeros(batch_size, 1).to(x.device)
        outputs = []

        for _ in range(self.max_steps):
            h, c = self.rnn(x, (h, c))
            y = torch.sigmoid(self.halting(h))
            halting_probability += y * (1 - halting_probability)
            remainders += (1 - halting_probability)
            n_updates += 1.0
            outputs.append(h.unsqueeze(1))
            
            if halting_probability.min() > 1 - 0.01:
                break

        outputs = torch.cat(outputs, dim=1)
        avg_outputs = torch.sum(outputs * remainders.unsqueeze(-1), dim=1) / n_updates
        return avg_outputs, halting_probability, n_updates

class HierarchicalExperts(nn.Module):
    def __init__(self, num_levels, num_experts_per_level, input_size, hidden_size, output_size):
        super().__init__()
        self.levels = nn.ModuleList([
            SparseMoE(num_experts_per_level, input_size if i == 0 else hidden_size, hidden_size, output_size if i == num_levels-1 else hidden_size)
            for i in range(num_levels)
        ])

    def forward(self, x):
        for level in self.levels:
            x = level(x)
        return x

class MoRALayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.hierarchical_experts = HierarchicalExperts(
            config.num_levels, 
            config.num_experts_per_level, 
            config.in_features, 
            config.expert_hidden_size, 
            config.out_features
        )
        self.act = AdaptiveComputationTime(config.in_features, config.expert_hidden_size, config.max_act_steps)
        self.layer_norm = nn.LayerNorm(config.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, halting_prob, n_updates = self.act(x)
        x = self.hierarchical_experts(x)
        return self.layer_norm(x)

class MoRALinear(nn.Linear):
    def __init__(self, config: Config):
        nn.Linear.__init__(self, config.in_features, config.out_features, bias=config.use_bias)
        self.mora = MoRALayer(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias) + self.mora(x)

class MoRAModel(nn.Module):
    def __init__(self, base_model: PreTrainedModel, config: Config):
        super().__init__()
        self.base_model = base_model
        self.config = config
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
                    MoRALinear(self.config)
                )

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

class LARS(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0, trust_coefficient=0.001):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                dp = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)

                momentum = state['momentum']
                weight_norm = torch.norm(p.data)
                grad_norm = torch.norm(dp)

                if weight_norm > 0 and grad_norm > 0:
                    local_lr = group['lr'] * group['trust_coefficient'] * weight_norm / grad_norm
                else:
                    local_lr = group['lr']

                if group['weight_decay'] != 0:
                    dp = dp.add(p.data, alpha=group['weight_decay'])

                momentum.mul_(group['momentum']).add_(dp, alpha=local_lr)
                p.data.add_(momentum, alpha=-1)

        return loss

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

class MoRAScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, config: Config):
        self.warmup_steps = config.warmup_steps
        self.total_steps = config.total_steps
        self.min_lr = config.min_lr
        self.cycle_length = config.cycle_length
        self.cycle_mult = config.cycle_mult
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cycle = math.floor(1 + progress * self.total_steps / self.cycle_length)
            x = abs(progress * self.total_steps / self.cycle_length - cycle + 1)
            cycle_factor = max(0, (1 - x) * self.cycle_mult ** (cycle - 1))
            return [max(self.min_lr, base_lr * (1 - progress) * cycle_factor) for base_lr in self.base_lrs]

def apply_mora(model: PreTrainedModel, config: Config) -> MoRAModel:
    return MoRAModel(model, config)

class TextClassificationHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class MoRAForSequenceClassification(nn.Module):
    def __init__(self, mora_model: MoRAModel, config: Config):
        super().__init__()
        self.mora_model = mora_model
        self.classifier = TextClassificationHead(config)

    def forward(self, input_ids, attention_mask):
        outputs = self.mora_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return logits

class AdvancedDataset(Dataset):
    def __init__(self, texts, labels, tokenizer: PreTrainedTokenizer, config: Config):
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = config.max_length
        self.encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def compute_metrics(preds, labels):
    preds = preds.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    cm = confusion_matrix(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }

def train_step(model, batch, optimizer, scheduler, scaler, device, config):
    model.train()
    batch = {k: v.to(device) for k, v in batch.items()}
    
    with autocast(enabled=config.use_mixed_precision):
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        loss = F.cross_entropy(outputs, batch['labels'])

    scaler.scale(loss).backward()
    
    if config.use_sam:
        optimizer.first_step(zero_grad=True)
        with autocast(enabled=config.use_mixed_precision):
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = F.cross_entropy(outputs, batch['labels'])
        scaler.scale(loss).backward()
        optimizer.second_step(zero_grad=True)
    else:
        scaler.unscale_(optimizer)
        if config.use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

    scheduler.step()

    return loss.item()

def eval_step(model, batch, device):
    model.eval()
    batch = {k: v.to(device) for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        loss = F.cross_entropy(outputs, batch['labels'])
    
    return loss.item(), outputs.cpu().numpy(), batch['labels'].cpu().numpy()

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, scaler, device, config):
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")):
            loss = train_step(model, batch, optimizer, scheduler, scaler, device, config)
            total_loss += loss

            if step % config.logging_steps == 0:
                logger.info(f"Epoch {epoch + 1}/{config.num_epochs} - Step {step} - Loss: {loss:.4f}")

            if step % config.eval_steps == 0:
                val_loss, val_metrics = evaluate(model, val_dataloader, device, config)
                logger.info(f"Validation Loss: {val_loss:.4f}")
                logger.info(f"Validation Metrics: {val_metrics}")

                if val_loss < best_val_loss - config.min_delta:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), config.best_model_path)
                    logger.info("New best model saved!")
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= config.early_stopping_patience:
                    logger.info("Early stopping triggered.")
                    return

            if step % config.checkpoint_interval == 0:
                torch.save(model.state_dict(), f"{config.checkpoint_path}_step_{step}.pth")

        avg_train_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs} - Average train loss: {avg_train_loss:.4f}")

        if config.save_every_epoch:
            torch.save(model.state_dict(), f"{config.model_path}_epoch_{epoch + 1}.pth")

    logger.info("Training completed.")

def evaluate(model, dataloader, device, config):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            loss, preds, labels = eval_step(model, batch, device)
            total_loss += loss
            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_preds), np.array(all_labels))

    return avg_loss, metrics

def create_data_loaders(dataset, config):
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split = int(np.floor(config.val_size * len(dataset)))
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=config.train_batch_size, num_workers=config.num_workers)
    val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=config.eval_batch_size, num_workers=config.num_workers)
    
    return train_loader, val_loader

def initialize_optimizer(model, config):
    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), **config.get_optimizer_params())
    elif config.optimizer == "LARS":
        optimizer = LARS(model.parameters(), **config.get_optimizer_params())
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    
    if config.use_sam:
        optimizer = SAM(model.parameters(), optimizer, **config.get_sam_params())
    
    return optimizer

def initialize_scheduler(optimizer, config):
    return MoRAScheduler(optimizer, config)

def main():
    # Load configuration
    config = Config('config.yaml')

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_mixed_precision = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7
    config.use_mixed_precision = use_mixed_precision

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Load pre-trained model and tokenizer
    base_model = BertModel.from_pretrained(config.base_model_name)
    tokenizer = BertTokenizer.from_pretrained(config.base_model_name)

    # Apply MoRA to the base model
    mora_model = apply_mora(base_model, config)

    # Create the classification model
    model = MoRAForSequenceClassification(mora_model, config).to(device)

    # Prepare dataset (replace with your actual dataset loading logic)
    texts, labels = load_dataset(config.dataset_path)
    dataset = AdvancedDataset(texts, labels, tokenizer, config)

    # Create data loaders
    train_dataloader, val_dataloader = create_data_loaders(dataset, config)

    # Initialize optimizer and scheduler
    optimizer = initialize_optimizer(model, config)
    scheduler = initialize_scheduler(optimizer, config)

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler(enabled=config.use_mixed_precision)

    # Train the model
    train(model, train_dataloader, val_dataloader, optimizer, scheduler, scaler, device, config)

    # Load best model and evaluate
    model.load_state_dict(torch.load(config.best_model_path))
    test_loss, test_metrics = evaluate(model, val_dataloader, device, config)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Metrics: {test_metrics}")

if __name__ == "__main__":
    main()