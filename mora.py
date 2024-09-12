import math
import logging
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from transformers import PreTrainedModel, PreTrainedTokenizer, BertModel, BertTokenizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OptimizerParams(BaseModel):
    lr: float
    momentum: Optional[float] = 0.9
    weight_decay: Optional[float] = 0.0
    trust_coefficient: Optional[float] = 0.001


class SamParams(BaseModel):
    rho: float = 0.05
    adaptive: bool = False


class SchedulerParams(BaseModel):
    warmup_steps: int
    total_steps: int
    min_lr: float
    cycle_length: int
    cycle_mult: int


class Config(BaseModel):
    lr: float
    num_epochs: int
    base_model_name: str
    num_experts: int
    expert_hidden_size: int
    optimizer: str
    optimizer_params: OptimizerParams
    use_sam: bool = False
    sam_params: Optional[SamParams] = None
    scheduler: str = "MoRAScheduler"
    scheduler_params: SchedulerParams
    use_mixed_precision: bool = True
    use_gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    train_batch_size: int
    eval_batch_size: int
    num_workers: int = 4
    val_size: float = 0.1
    logging_steps: int = 50
    eval_steps: int = 200
    checkpoint_interval: int = 500
    checkpoint_path: str = "./checkpoints/model"
    best_model_path: str = "./checkpoints/best_model.pth"
    early_stopping_patience: int = 3
    min_delta: float = 0.001
    save_every_epoch: bool = True
    model_path: str = "./checkpoints/model_epoch.pth"
    dataset_path: str
    max_length: int = 128
    classifier_dropout: float = 0.1
    num_labels: int = 2
    num_levels: int = 2
    num_experts_per_level: int = 4
    in_features: int = 768
    out_features: int = 768
    max_act_steps: int = 10

    @validator("sam_params", always=True)
    def check_sam_params(cls, v, values):
        if values.get("use_sam") and v is None:
            raise ValueError("sam_params must be provided if use_sam is True")
        return v


def load_config(config_path: str) -> Config:
    """
    Load and validate configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Config: Validated configuration object.
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)


class DynamicExpert(nn.Module):
    """
    A simple two-layer neural network with ReLU activation.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        return self.fc2(x)


class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts (MoE) layer that selects top-k experts based on gate logits.
    """

    def __init__(self, num_experts, input_size, hidden_size, output_size, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList(
            [DynamicExpert(input_size, hidden_size, output_size) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(input_size, num_experts)

    def forward(self, x):
        gate_logits = self.gate(x)
        weights, indices = torch.topk(gate_logits, self.k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        batch_size = x.size(0)
        expert_outputs = []
        for i in range(self.k):
            expert = self.experts[indices[:, i]]
            expert_output = expert(x)
            expert_outputs.append(expert_output)

        expert_outputs = torch.stack(expert_outputs, dim=1)  # Shape: (batch_size, k, output_size)
        output = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=1)
        return output


class AdaptiveComputationTime(nn.Module):
    """
    Adaptive Computation Time (ACT) module to dynamically decide the number of computational steps.
    """

    def __init__(self, input_size, hidden_size, max_steps):
        super().__init__()
        self.max_steps = max_steps
        self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.halting = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h, c = (
            torch.zeros(batch_size, self.rnn.hidden_size).to(x.device),
            torch.zeros(batch_size, self.rnn.hidden_size).to(x.device),
        )
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
    """
    Hierarchical Experts model consisting of multiple levels of SparseMoE layers.
    """

    def __init__(
        self,
        num_levels,
        num_experts_per_level,
        input_size,
        hidden_size,
        output_size,
    ):
        super().__init__()
        self.levels = nn.ModuleList(
            [
                SparseMoE(
                    num_experts_per_level,
                    input_size if i == 0 else hidden_size,
                    hidden_size,
                    output_size if i == num_levels - 1 else hidden_size,
                )
                for i in range(num_levels)
            ]
        )

    def forward(self, x):
        for level in self.levels:
            x = level(x)
        return x


class MoRALayer(nn.Module):
    """
    MoRA Layer that integrates HierarchicalExperts and AdaptiveComputationTime.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.hierarchical_experts = HierarchicalExperts(
            num_levels=config.num_levels,
            num_experts_per_level=config.num_experts_per_level,
            input_size=config.in_features,
            hidden_size=config.expert_hidden_size,
            output_size=config.out_features,
        )
        self.act = AdaptiveComputationTime(
            input_size=config.in_features,
            hidden_size=config.expert_hidden_size,
            max_steps=config.max_act_steps,
        )
        self.layer_norm = nn.LayerNorm(config.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, halting_prob, n_updates = self.act(x)
        x = self.hierarchical_experts(x)
        return self.layer_norm(x)


class MoRALinear(nn.Linear):
    """
    Linear layer augmented with MoRALayer.
    """

    def __init__(self, config: Config):
        super().__init__(config.in_features, config.out_features, bias=True)
        self.mora = MoRALayer(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias) + self.mora(x)


class MoRAModel(nn.Module):
    """
    Model wrapper that replaces linear layers in the base model with MoRALinear.
    """

    def __init__(self, base_model: PreTrainedModel, config: Config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self._replace_linear_layers()

    def _replace_linear_layers(self):
        """
        Replace all nn.Linear layers in the base model with MoRALinear.
        """
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                parent = self.base_model
                attrs = name.split(".")
                for attr in attrs[:-1]:
                    parent = getattr(parent, attr)
                linear_layer = MoRALinear(self.config)
                setattr(parent, attrs[-1], linear_layer)
                logger.info(f"Replaced {name} with MoRALinear.")

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)


class TextClassificationHead(nn.Module):
    """
    Classification head for sequence classification tasks.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.dense = nn.Linear(config.out_features, config.out_features)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.out_features, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MoRAForSequenceClassification(nn.Module):
    """
    Complete model integrating MoRAModel and a classification head.
    """

    def __init__(self, mora_model: MoRAModel, config: Config):
        super().__init__()
        self.mora_model = mora_model
        self.classifier = TextClassificationHead(config)

    def forward(self, input_ids, attention_mask):
        outputs = self.mora_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Assuming BERT-like pooled output
        logits = self.classifier(pooled_output)
        return logits


class AdvancedDataset(Dataset):
    """
    Custom dataset for text classification tasks.
    """

    def __init__(self, texts, labels, tokenizer: PreTrainedTokenizer, config: Config):
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = config.max_length
        self.encodings = self.tokenizer(
            texts, truncation=True, padding=True, max_length=self.max_length
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(preds, labels):
    """
    Compute evaluation metrics.

    Args:
        preds (np.ndarray): Predicted logits.
        labels (np.ndarray): True labels.

    Returns:
        Dict[str, Any]: Dictionary of computed metrics.
    """
    preds = preds.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    try:
        roc_auc = roc_auc_score(labels, preds, average="weighted", multi_class="ovr")
    except ValueError:
        roc_auc = float("nan")
    cm = confusion_matrix(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
    }


def train_step(
    model,
    batch,
    optimizer,
    scheduler,
    scaler,
    device,
    config: Config,
):
    """
    Perform a single training step.

    Args:
        model (nn.Module): The model to train.
        batch (Dict[str, torch.Tensor]): Batch of data.
        optimizer (Optimizer): Optimizer.
        scheduler (_LRScheduler): Learning rate scheduler.
        scaler (GradScaler): Gradient scaler for mixed precision.
        device (torch.device): Device to perform computations on.
        config (Config): Configuration object.

    Returns:
        float: Loss value.
    """
    model.train()
    batch = {k: v.to(device) for k, v in batch.items()}

    optimizer.zero_grad()
    with autocast(enabled=config.use_mixed_precision):
        outputs = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        loss = F.cross_entropy(outputs, batch["labels"])

    scaler.scale(loss).backward()

    if config.use_gradient_clipping:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    return loss.item()


def eval_step(model, batch, device):
    """
    Perform a single evaluation step.

    Args:
        model (nn.Module): The model to evaluate.
        batch (Dict[str, torch.Tensor]): Batch of data.
        device (torch.device): Device to perform computations on.

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: Loss, predictions, and true labels.
    """
    model.eval()
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = F.cross_entropy(outputs, batch["labels"])
        logits = outputs.detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()

    return loss.item(), logits, labels


def evaluate(model, dataloader, device, config: Config):
    """
    Evaluate the model on the given dataloader.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Dataloader for evaluation data.
        device (torch.device): Device to perform computations on.
        config (Config): Configuration object.

    Returns:
        Tuple[float, Dict[str, Any]]: Average loss and computed metrics.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            loss, preds, labels = eval_step(model, batch, device)
            total_loss += loss
            all_preds.append(preds)
            all_labels.append(labels)

    avg_loss = total_loss / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(all_preds, all_labels)

    return avg_loss, metrics


def create_data_loaders(dataset, config: Config):
    """
    Create training and validation dataloaders with stratified splitting.

    Args:
        dataset (Dataset): The dataset to split.
        config (Config): Configuration object.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation dataloaders.
    """
    labels = np.array(dataset.labels)
    train_indices, val_indices = train_test_split(
        np.arange(len(labels)),
        test_size=config.val_size,
        stratify=labels,
        random_state=42,
    )

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=config.train_batch_size,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        dataset,
        sampler=val_sampler,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
    )

    return train_loader, val_loader


def initialize_optimizer(model, config: Config) -> Optimizer:
    """
    Initialize the optimizer based on the configuration.

    Args:
        model (nn.Module): The model parameters to optimize.
        config (Config): Configuration object.

    Returns:
        Optimizer: Initialized optimizer.
    """
    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), **config.optimizer_params.dict())
    elif config.optimizer == "LARS":
        optimizer = LARS(model.parameters(), **config.optimizer_params.dict())
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    if config.use_sam:
        optimizer = SAM(model.parameters(), optimizer, **config.sam_params.dict())

    return optimizer


def initialize_scheduler(optimizer, config: Config) -> _LRScheduler:
    """
    Initialize the learning rate scheduler based on the configuration.

    Args:
        optimizer (Optimizer): Optimizer to schedule.
        config (Config): Configuration object.

    Returns:
        _LRScheduler: Initialized scheduler.
    """
    return MoRAScheduler(optimizer, config)


class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS) optimizer.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.9,
        weight_decay=0,
        trust_coefficient=0.001,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "trust_coefficient": trust_coefficient,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            trust_coefficient = group["trust_coefficient"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                param_norm = p.data.norm(2)
                grad_norm = d_p.norm(2)

                if param_norm != 0 and grad_norm != 0:
                    local_lr = trust_coefficient * param_norm / (grad_norm + 1e-9)
                else:
                    local_lr = 1.0

                d_p = d_p.mul(local_lr)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    buf = state["momentum_buffer"] = torch.clone(d_p).detach()
                else:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(d_p)

                p.data.add_(-lr, buf)

        return loss


class SAM(Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.
    """

    def __init__(
        self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs
    ):
        assert rho >= 0.0, "Invalid rho, should be non-negative"
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
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # ascent step
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # descent step

        self.base_optimizer.step()  # update step

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("SAM requires a closure.")

        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step(zero_grad=True)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(2)
                for group in self.param_groups for p in group["params"] if p.grad is not None
            ]),
            p=2,
        )
        return norm


class MoRAScheduler(_LRScheduler):
    """
    Custom learning rate scheduler for MoRA.
    """

    def __init__(self, optimizer: Optimizer, config: Config):
        self.warmup_steps = config.scheduler_params.warmup_steps
        self.total_steps = config.scheduler_params.total_steps
        self.min_lr = config.scheduler_params.min_lr
        self.cycle_length = config.scheduler_params.cycle_length
        self.cycle_mult = config.scheduler_params.cycle_mult
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                base_lr * (self.last_epoch / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            progress = (self.last_epoch - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            cycle = math.floor(1 + progress * self.total_steps / self.cycle_length)
            x = abs(progress * self.total_steps / self.cycle_length - cycle + 1)
            cycle_factor = max(0, (1 - x) * self.cycle_mult ** (cycle - 1))
            return [
                max(
                    self.min_lr,
                    base_lr * (1 - progress) * cycle_factor,
                )
                for base_lr in self.base_lrs
            ]


def apply_mora(model: PreTrainedModel, config: Config) -> MoRAModel:
    """
    Apply MoRA modifications to the base model.

    Args:
        model (PreTrainedModel): The pre-trained base model.
        config (Config): Configuration object.

    Returns:
        MoRAModel: Modified model with MoRA layers.
    """
    return MoRAModel(model, config)


def load_dataset(dataset_path: str):
    """
    Load dataset from a CSV file.

    Args:
        dataset_path (str): Path to the CSV file.

    Returns:
        Tuple[List[str], List[int]]: Texts and corresponding labels.
    """
    import pandas as pd

    df = pd.read_csv(dataset_path)
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    return texts, labels


def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    scaler,
    device,
    config: Config,
):
    """
    Train the model with early stopping and checkpointing.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): Dataloader for training data.
        val_dataloader (DataLoader): Dataloader for validation data.
        optimizer (Optimizer): Optimizer.
        scheduler (_LRScheduler): Learning rate scheduler.
        scaler (GradScaler): Gradient scaler for mixed precision.
        device (torch.device): Device to perform computations on.
        config (Config): Configuration object.
    """
    best_val_loss = float("inf")
    early_stopping_counter = 0

    try:
        for epoch in range(config.num_epochs):
            model.train()
            total_loss = 0
            for step, batch in enumerate(
                tqdm(
                    train_dataloader,
                    desc=f"Epoch {epoch + 1}/{config.num_epochs}",
                )
            ):
                loss = train_step(
                    model, batch, optimizer, scheduler, scaler, device, config
                )
                total_loss += loss

                if (step + 1) % config.logging_steps == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{config.num_epochs} - Step {step + 1} - Loss: {loss:.4f}"
                    )

                if (step + 1) % config.eval_steps == 0:
                    val_loss, val_metrics = evaluate(
                        model, val_dataloader, device, config
                    )
                    logger.info(f"Validation Loss: {val_loss:.4f}")
                    logger.info(f"Validation Metrics: {val_metrics}")

                    if val_loss < best_val_loss - config.min_delta:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), config.best_model_path)
                        logger.info("New best model saved!")
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                        logger.info(
                            f"No improvement in validation loss. Early stopping counter: {early_stopping_counter}"
                        )

                    if early_stopping_counter >= config.early_stopping_patience:
                        logger.info("Early stopping triggered.")
                        return

                if (step + 1) % config.checkpoint_interval == 0:
                    checkpoint_file = f"{config.checkpoint_path}_step_{step + 1}.pth"
                    torch.save(model.state_dict(), checkpoint_file)
                    logger.info(f"Checkpoint saved at step {step + 1}.")

            avg_train_loss = total_loss / len(train_dataloader)
            logger.info(
                f"Epoch {epoch + 1}/{config.num_epochs} - Average Train Loss: {avg_train_loss:.4f}"
            )

            if config.save_every_epoch:
                epoch_model_path = f"{config.checkpoint_path}_epoch_{epoch + 1}.pth"
                torch.save(model.state_dict(), epoch_model_path)
                logger.info(f"Model saved for epoch {epoch + 1}.")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")

    finally:
        logger.info("Training completed.")


def main():
    """
    Main function to orchestrate the training and evaluation pipeline.
    """
    try:
        # Load configuration
        config = load_config("config.yaml")
        logger.info("Configuration loaded successfully.")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load pre-trained model and tokenizer
        base_model = BertModel.from_pretrained(config.base_model_name)
        tokenizer = BertTokenizer.from_pretrained(config.base_model_name)
        logger.info(f"Loaded base model: {config.base_model_name}")

        # Apply MoRA to the base model
        mora_model = apply_mora(base_model, config)
        logger.info("Applied MoRA to the base model.")

        # Create the classification model
        model = MoRAForSequenceClassification(mora_model, config).to(device)
        logger.info("Initialized MoRAForSequenceClassification model.")

        # Prepare dataset
        texts, labels = load_dataset(config.dataset_path)
        dataset = AdvancedDataset(texts, labels, tokenizer, config)
        logger.info("Dataset loaded and tokenized.")

        # Create data loaders
        train_dataloader, val_dataloader = create_data_loaders(dataset, config)
        logger.info("Data loaders created.")

        # Initialize optimizer and scheduler
        optimizer = initialize_optimizer(model, config)
        scheduler = initialize_scheduler(optimizer, config)
        logger.info("Optimizer and scheduler initialized.")

        # Initialize gradient scaler for mixed precision training
        scaler = GradScaler(enabled=config.use_mixed_precision)
        logger.info("Gradient scaler initialized.")

        # Train the model
        train(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            scheduler,
            scaler,
            device,
            config,
        )

        # Load best model and evaluate
        model.load_state_dict(torch.load(config.best_model_path))
        logger.info("Best model loaded for evaluation.")

        test_loss, test_metrics = evaluate(model, val_dataloader, device, config)
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Metrics: {test_metrics}")

    except Exception as e:
        logger.error(f"An error occurred in the main pipeline: {e}")


if __name__ == "__main__":
    main()
