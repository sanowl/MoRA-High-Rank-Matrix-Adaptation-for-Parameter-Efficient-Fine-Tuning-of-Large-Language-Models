import math
import logging
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import PreTrainedModel, PreTrainedTokenizer, BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
  confusion_matrix,
  accuracy_score,
  precision_score,
  recall_score,
  f1_score,
  roc_auc_score,
  matthews_corrcoef,
  classification_report,
)
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, Tuple, List
import random
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
  level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class OptimizerParams(BaseModel):
  lr: float
  momentum: Optional[float] = 0.9
  weight_decay: Optional[float] = 0.01
  betas: Optional[Tuple[float, float]] = (0.9, 0.999)

class SamParams(BaseModel):
  rho: float = 0.05
  adaptive: bool = False

class SchedulerParams(BaseModel):
  max_lr: float
  total_steps: int
  anneal_strategy: str = "cos"
  pct_start: float = 0.3
  div_factor: float = 25.0
  final_div_factor: float = 1e4

class MoraParams(BaseModel):
  num_levels: int
  num_experts_per_level: int
  expert_hidden_size: int
  in_features: int
  out_features: int
  max_act_steps: int

class Config(BaseModel):
  num_epochs: int
  train_batch_size: int
  eval_batch_size: int
  learning_rate: float
  optimizer: str
  optimizer_params: OptimizerParams
  scheduler: str
  scheduler_params: SchedulerParams
  weight_decay: float = 0.01
  max_grad_norm: float = 1.0
  early_stopping_patience: int = 3
  gradient_accumulation_steps: int = 1
  mixed_precision: bool = True
  seed: int = 42
  base_model_name: str
  num_labels: int = 2
  classifier_dropout: float = 0.1
  mora: MoraParams
  dataset_path: str
  max_length: int = 128
  val_size: float = 0.1
  num_workers: int = 4
  checkpoint_dir: str = "./checkpoints"
  best_model_name: str = "best_model.pth"
  checkpoint_interval: int = 500
  save_every_epoch: bool = True
  logging_steps: int = 50
  eval_steps: int = 200
  use_sam: bool = False
  sam_params: Optional[SamParams] = None

  @validator("sam_params", always=True)
  def check_sam_params(cls, v, values):
    if values.get("use_sam") and v is None:
      raise ValueError("sam_params must be provided if use_sam is True")
    return v

def set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  logger.info(f"Random seed set to {seed}")

def load_config(config_path: str) -> Config:
  with open(config_path, "r") as f:
    config_dict = yaml.safe_load(f)
  return Config(**config_dict)

class DynamicExpert(nn.Module):
  def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
    super().__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_size)
    self.act = nn.ReLU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.act(self.fc1(x))
    return self.fc2(x)

class SparseMoE(nn.Module):
  def __init__(
    self,
    num_experts: int,
    input_size: int,
    hidden_size: int,
    output_size: int,
    k: int = 2,
  ) -> None:
    super().__init__()
    self.num_experts = num_experts
    self.k = k
    self.experts = nn.ModuleList(
      [DynamicExpert(input_size, hidden_size, output_size) for _ in range(num_experts)]
    )
    self.gate = nn.Linear(input_size, num_experts)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate_logits = self.gate(x)
    weights, indices = torch.topk(gate_logits, self.k, dim=-1)
    weights = F.softmax(weights, dim=-1)
    batch_size = x.size(0)
    expert_outputs: List[torch.Tensor] = []
    for i in range(self.k):
      selected_experts = self.experts[indices[:, i]]
      expert_output = selected_experts(x)
      expert_outputs.append(expert_output.unsqueeze(1))
    expert_outputs_tensor = torch.cat(expert_outputs, dim=1)
    output = torch.sum(weights.unsqueeze(-1) * expert_outputs_tensor, dim=1)
    return output

class AdaptiveComputationTime(nn.Module):
  def __init__(self, input_size: int, hidden_size: int, max_steps: int) -> None:
    super().__init__()
    self.max_steps = max_steps
    self.rnn = nn.LSTMCell(input_size, hidden_size)
    self.halting = nn.Linear(hidden_size, 1)

  def forward(
    self, x: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = x.size(0)
    h, c = (
      torch.zeros(batch_size, self.rnn.hidden_size, device=x.device),
      torch.zeros(batch_size, self.rnn.hidden_size, device=x.device),
    )
    halting_probability = torch.zeros(batch_size, 1, device=x.device)
    remainders = torch.zeros(batch_size, 1, device=x.device)
    n_updates = torch.zeros(batch_size, 1, device=x.device)
    outputs: List[torch.Tensor] = []
    for _ in range(self.max_steps):
      h, c = self.rnn(x, (h, c))
      y = torch.sigmoid(self.halting(h))
      halting_probability += y * (1 - halting_probability)
      remainders += (1 - halting_probability)
      n_updates += 1.0
      outputs.append(h.unsqueeze(1))
      if halting_probability.min() > 1 - 0.01:
        break
    outputs_tensor = torch.cat(outputs, dim=1)
    avg_outputs = torch.sum(outputs_tensor * remainders.unsqueeze(-1), dim=1) / n_updates
    return avg_outputs, halting_probability, n_updates

class HierarchicalExperts(nn.Module):
  def __init__(
    self,
    num_levels: int,
    num_experts_per_level: int,
    input_size: int,
    hidden_size: int,
    output_size: int,
  ) -> None:
    super().__init__()
    self.levels = nn.ModuleList(
      [
        SparseMoE(
          num_experts=num_experts_per_level,
          input_size=input_size if i == 0 else hidden_size,
          hidden_size=hidden_size,
          output_size=output_size if i == num_levels - 1 else hidden_size,
        )
        for i in range(num_levels)
      ]
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for level in self.levels:
      x = level(x)
    return x

class MoRALayer(nn.Module):
  def __init__(self, config: Config) -> None:
    super().__init__()
    self.hierarchical_experts = HierarchicalExperts(
      num_levels=config.mora.num_levels,
      num_experts_per_level=config.mora.num_experts_per_level,
      input_size=config.mora.in_features,
      hidden_size=config.mora.expert_hidden_size,
      output_size=config.mora.out_features,
    )
    self.act = AdaptiveComputationTime(
      input_size=config.mora.in_features,
      hidden_size=config.mora.expert_hidden_size,
      max_steps=config.mora.max_act_steps,
    )
    self.layer_norm = nn.LayerNorm(config.mora.out_features)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x, halting_prob, n_updates = self.act(x)
    x = self.hierarchical_experts(x)
    return self.layer_norm(x)

class MoRALinear(nn.Linear):
  def __init__(self, config: Config) -> None:
    super().__init__(config.mora.in_features, config.mora.out_features, bias=True)
    self.mora = MoRALayer(config)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, self.weight, self.bias) + self.mora(x)

class MoRAModel(nn.Module):
  def __init__(self, base_model: PreTrainedModel, config: Config) -> None:
    super().__init__()
    self.base_model = base_model
    self.config = config
    self._replace_linear_layers()

  def _replace_linear_layers(self) -> None:
    for name, module in self.base_model.named_modules():
      if isinstance(module, nn.Linear):
        parent = self.base_model
        attrs = name.split(".")
        for attr in attrs[:-1]:
          parent = getattr(parent, attr)
        linear_layer = MoRALinear(self.config)
        setattr(parent, attrs[-1], linear_layer)
        logger.info(f"Replaced {name} with MoRALinear.")

  def forward(self, *args: Any, **kwargs: Any) -> Any:
    return self.base_model(*args, **kwargs)

class TextClassificationHead(nn.Module):
  def __init__(self, config: Config) -> None:
    super().__init__()
    self.dense = nn.Linear(config.mora.out_features, config.mora.out_features)
    self.dropout = nn.Dropout(config.classifier_dropout)
    self.out_proj = nn.Linear(config.mora.out_features, config.num_labels)

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    x = self.dropout(features)
    x = self.dense(x)
    x = torch.tanh(x)
    x = self.dropout(x)
    x = self.out_proj(x)
    return x

class MoRAForSequenceClassification(nn.Module):
  def __init__(self, mora_model: MoRAModel, config: Config) -> None:
    super().__init__()
    self.mora_model = mora_model
    self.classifier = TextClassificationHead(config)

  def forward(
    self, input_ids: torch.Tensor, attention_mask: torch.Tensor
  ) -> torch.Tensor:
    outputs = self.mora_model(input_ids=input_ids, attention_mask=attention_mask)
    pooled_output = outputs.last_hidden_state[:, 0]
    logits = self.classifier(pooled_output)
    return logits

class AdvancedDataset(Dataset):
  def __init__(
    self,
    texts: List[str],
    labels: List[int],
    tokenizer: PreTrainedTokenizer,
    config: Config,
  ) -> None:
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_length = config.max_length
    self.encodings = self.tokenizer(
      texts,
      truncation=True,
      padding=True,
      max_length=self.max_length,
      return_tensors="pt",
    )

  def __len__(self) -> int:
    return len(self.labels)

  def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    item = {key: val[idx] for key, val in self.encodings.items()}
    item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
    return item

def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
  preds = preds.argmax(axis=1)
  acc = accuracy_score(labels, preds)
  precision = precision_score(labels, preds, average="weighted", zero_division=0)
  recall = recall_score(labels, preds, average="weighted", zero_division=0)
  f1 = f1_score(labels, preds, average="weighted", zero_division=0)
  mcc = matthews_corrcoef(labels, preds)
  try:
    roc_auc = roc_auc_score(labels, preds, average="weighted", multi_class="ovr")
  except ValueError:
    roc_auc = float("nan")
  cm = confusion_matrix(labels, preds)
  report = classification_report(labels, preds, zero_division=0, output_dict=True)
  return {
    "accuracy": acc,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "mcc": mcc,
    "roc_auc": roc_auc,
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
  }

class LARS(Optimizer):
  def __init__(
    self,
    params: Any,
    lr: float = 1e-3,
    momentum: float = 0.9,
    weight_decay: float = 0,
    trust_coefficient: float = 0.001,
  ) -> None:
    defaults = {
      "lr": lr,
      "momentum": momentum,
      "weight_decay": weight_decay,
      "trust_coefficient": trust_coefficient,
    }
    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure: Optional[Any] = None) -> Optional[float]:
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
  def __init__(
    self,
    params: Any,
    base_optimizer: Any,
    rho: float = 0.05,
    adaptive: bool = False,
    **kwargs: Any,
  ) -> None:
    assert rho >= 0.0, "Invalid rho, should be non-negative"
    defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
    super(SAM, self).__init__(params, defaults)
    self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
    self.param_groups = self.base_optimizer.param_groups

  @torch.no_grad()
  def first_step(self, zero_grad: bool = False) -> None:
    grad_norm = self._grad_norm()
    for group in self.param_groups:
      scale = group["rho"] / (grad_norm + 1e-12)
      for p in group["params"]:
        if p.grad is None:
          continue
        e_w = (
          (torch.pow(p, 2) if group["adaptive"] else 1.0)
          * p.grad
          * scale
        ).to(p)
        p.add_(e_w)
        self.state[p]["e_w"] = e_w
    if zero_grad:
      self.zero_grad()

  @torch.no_grad()
  def second_step(self, zero_grad: bool = False) -> None:
    for group in self.param_groups:
      for p in group["params"]:
        if p.grad is None:
          continue
        p.sub_(self.state[p]["e_w"])
    self.base_optimizer.step()
    if zero_grad:
      self.zero_grad()

  @torch.no_grad()
  def step(self, closure: Optional[Any] = None) -> Any:
    if closure is None:
      raise ValueError("SAM requires a closure.")
    closure = torch.enable_grad()(closure)
    self.first_step(zero_grad=True)
    closure()
    self.second_step(zero_grad=True)

  def _grad_norm(self) -> torch.Tensor:
    shared_device = self.param_groups[0]["params"][0].device
    norm = torch.norm(
      torch.stack([
        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(2)
        for group in self.param_groups for p in group["params"] if p.grad is not None
      ]),
      p=2,
    )
    return norm

def train_step(
  model: nn.Module,
  batch: Dict[str, torch.Tensor],
  optimizer: Optimizer,
  scaler: GradScaler,
  device: torch.device,
  config: Config,
) -> float:
  model.train()
  batch = {k: v.to(device) for k, v in batch.items()}
  optimizer.zero_grad()
  with autocast(enabled=config.mixed_precision):
    outputs = model(
      input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    loss = F.cross_entropy(outputs, batch["labels"])
  scaler.scale(loss).backward()
  if config.max_grad_norm > 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
  scaler.step(optimizer)
  scaler.update()
  return loss.item()

def eval_step(
  model: nn.Module, batch: Dict[str, torch.Tensor], device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
  model.eval()
  batch = {k: v.to(device) for k, v in batch.items()}
  with torch.no_grad():
    outputs = model(
      input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    loss = F.cross_entropy(outputs, batch["labels"])
    logits = outputs.detach().cpu().numpy()
    labels = batch["labels"].detach().cpu().numpy()
  return loss.item(), logits, labels

def evaluate(
  model: nn.Module,
  dataloader: DataLoader,
  device: torch.device,
  config: Config,
  writer: Optional[SummaryWriter] = None,
  global_step: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
  model.eval()
  total_loss = 0.0
  all_preds: List[np.ndarray] = []
  all_labels: List[np.ndarray] = []
  with torch.no_grad():
    for batch in tqdm(dataloader, desc="Evaluating"):
      loss, preds, labels = eval_step(model, batch, device)
      total_loss += loss
      all_preds.append(preds)
      all_labels.append(labels)
  avg_loss = total_loss / len(dataloader)
  all_preds_np = np.concatenate(all_preds, axis=0)
  all_labels_np = np.concatenate(all_labels, axis=0)
  metrics = compute_metrics(all_preds_np, all_labels_np)
  if writer and global_step is not None:
    writer.add_scalar("Eval/Loss", avg_loss, global_step)
    writer.add_scalar("Eval/Accuracy", metrics["accuracy"], global_step)
    writer.add_scalar("Eval/F1", metrics["f1"], global_step)
    writer.add_scalar("Eval/MCC", metrics["mcc"], global_step)
  return avg_loss, metrics

def create_data_loaders(
  dataset: Dataset, config: Config
) -> Tuple[DataLoader, DataLoader]:
  labels = np.array(dataset.labels)
  train_size = int((1 - config.val_size) * len(labels))
  val_size = len(labels) - train_size
  train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(config.seed)
  )
  train_loader = DataLoader(
    train_dataset,
    batch_size=config.train_batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=True,
  )
  val_loader = DataLoader(
    val_dataset,
    batch_size=config.eval_batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=True,
  )
  return train_loader, val_loader

def initialize_optimizer_scheduler(
  model: nn.Module, config: Config
) -> Tuple[Optimizer, Any]:
  if config.optimizer == "AdamW":
    optimizer = AdamW(
      model.parameters(),
      lr=config.learning_rate,
      betas=config.optimizer_params.betas,
      weight_decay=config.weight_decay,
    )
  elif config.optimizer == "LARS":
    optimizer = LARS(
      model.parameters(),
      lr=config.learning_rate,
      momentum=config.optimizer_params.momentum,
      weight_decay=config.weight_decay,
      trust_coefficient=config.optimizer_params.trust_coefficient,
    )
  else:
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")
  if config.use_sam:
    optimizer = SAM(
      model.parameters(),
      base_optimizer=AdamW,
      rho=config.sam_params.rho,
      adaptive=config.sam_params.adaptive,
      lr=config.learning_rate,
      weight_decay=config.weight_decay,
    )
  scheduler = OneCycleLR(
    optimizer,
    max_lr=config.scheduler_params.max_lr,
    total_steps=config.scheduler_params.total_steps,
    anneal_strategy=config.scheduler_params.anneal_strategy,
    pct_start=config.scheduler_params.pct_start,
    div_factor=config.scheduler_params.div_factor,
    final_div_factor=config.scheduler_params.final_div_factor,
  )
  return optimizer, scheduler

def save_checkpoint(
  model: nn.Module,
  optimizer: Optimizer,
  scheduler: Any,
  scaler: GradScaler,
  epoch: int,
  step: int,
  path: str,
) -> None:
  state = {
    "epoch": epoch,
    "step": step,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict(),
  }
  torch.save(state, path)
  logger.info(f"Checkpoint saved at {path}")

def load_checkpoint(
  model: nn.Module,
  optimizer: Optimizer,
  scheduler: Any,
  scaler: GradScaler,
  path: str,
  device: torch.device,
) -> Tuple[int, int]:
  if not os.path.exists(path):
    raise FileNotFoundError(f"Checkpoint not found at {path}")
  checkpoint = torch.load(path, map_location=device)
  model.load_state_dict(checkpoint["model_state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
  scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
  scaler.load_state_dict(checkpoint["scaler_state_dict"])
  epoch = checkpoint["epoch"]
  step = checkpoint["step"]
  logger.info(f"Checkpoint loaded from {path} at epoch {epoch}, step {step}")
  return epoch, step

def train(
  model: nn.Module,
  train_dataloader: DataLoader,
  val_dataloader: DataLoader,
  optimizer: Optimizer,
  scheduler: Any,
  scaler: GradScaler,
  device: torch.device,
  config: Config,
  writer: SummaryWriter,
) -> None:
  best_val_loss = float("inf")
  early_stopping_counter = 0
  global_step = 0
  os.makedirs(config.checkpoint_dir, exist_ok=True)
  for epoch in range(1, config.num_epochs + 1):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{config.num_epochs}")):
      loss = train_step(model, batch, optimizer, scaler, device, config)
      total_loss += loss
      global_step += 1
      if (step + 1) % config.logging_steps == 0:
        avg_loss = total_loss / config.logging_steps
        logger.info(
          f"Epoch [{epoch}/{config.num_epochs}] Step [{step+1}/{len(train_dataloader)}] Loss: {avg_loss:.4f}"
        )
        writer.add_scalar("Train/Loss", avg_loss, global_step)
        total_loss = 0.0
      if (step + 1) % config.eval_steps == 0:
        val_loss, val_metrics = evaluate(model, val_dataloader, device, config, writer, global_step)
        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info(f"Validation Metrics: {val_metrics}")
        if val_loss < best_val_loss - config.scheduler_params.final_div_factor:
          best_val_loss = val_loss
          best_model_path = os.path.join(config.checkpoint_dir, config.best_model_name)
          torch.save(model.state_dict(), best_model_path)
          logger.info("New best model saved!")
          early_stopping_counter = 0
        else:
          early_stopping_counter += 1
          logger.info(f"No improvement in validation loss. Early stopping counter: {early_stopping_counter}")
        if early_stopping_counter >= config.early_stopping_patience:
          logger.info("Early stopping triggered.")
          return
        scheduler.step()
      if (step + 1) % config.checkpoint_interval == 0:
        checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{step+1}.pth")
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, step+1, checkpoint_path)
    avg_epoch_loss = total_loss / len(train_dataloader)
    logger.info(f"Epoch [{epoch}/{config.num_epochs}] Average Loss: {avg_epoch_loss:.4f}")
    writer.add_scalar("Train/Epoch_Loss", avg_epoch_loss, epoch)
    if config.save_every_epoch:
      epoch_checkpoint = os.path.join(config.checkpoint_dir, f"epoch_{epoch}.pth")
      save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, epoch_checkpoint)
  logger.info("Training completed.")

def load_dataset(dataset_path: str) -> Tuple[List[str], List[int]]:
  df = pd.read_csv(dataset_path)
  if 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError("Dataset must contain 'text' and 'label' columns.")
  texts = df["text"].astype(str).tolist()
  labels = df["label"].astype(int).tolist()
  return texts, labels

def main() -> None:
  try:
    config = load_config("config.yaml")
    logger.info("Configuration loaded successfully.")
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    log_dir = os.path.join("runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logging at {log_dir}")
    tokenizer = BertTokenizer.from_pretrained(config.base_model_name)
    base_model = BertModel.from_pretrained(config.base_model_name)
    logger.info(f"Loaded base model: {config.base_model_name}")
    mora_model = MoRAModel(base_model, config)
    logger.info("Applied MoRA to the base model.")
    model = MoRAForSequenceClassification(mora_model, config).to(device)
    logger.info("Initialized MoRAForSequenceClassification model.")
    texts, labels = load_dataset(config.dataset_path)
    dataset = AdvancedDataset(texts, labels, tokenizer, config)
    logger.info("Dataset loaded and tokenized.")
    train_dataloader, val_dataloader = create_data_loaders(dataset, config)
    logger.info("Data loaders created.")
    optimizer, scheduler = initialize_optimizer_scheduler(model, config)
    logger.info("Optimizer and scheduler initialized.")
    scaler = GradScaler(enabled=config.mixed_precision)
    logger.info("Gradient scaler initialized.")
    train(
      model,
      train_dataloader,
      val_dataloader,
      optimizer,
      scheduler,
      scaler,
      device,
      config,
      writer,
    )
    best_model_path = os.path.join(config.checkpoint_dir, config.best_model_name)
    if os.path.exists(best_model_path):
      model.load_state_dict(torch.load(best_model_path))
      logger.info("Best model loaded for evaluation.")
    else:
      logger.warning("Best model checkpoint not found. Using the last checkpoint.")
    test_loss, test_metrics = evaluate(model, val_dataloader, device, config, writer)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Metrics: {test_metrics}")
    report = classification_report(
      np.concatenate([batch["labels"].numpy() for batch in val_dataloader]),
      np.argmax(np.concatenate([batch["labels"].numpy() for batch in val_dataloader]), axis=1),
      zero_division=0,
    )
    writer.add_text("Eval/Classification_Report", report, 0)
    logger.info("Classification report logged.")
  except Exception as e:
    logger.error(f"An error occurred in the main pipeline: {e}")
  finally:
    logger.info("Training pipeline finished.")
    writer.close()

if __name__ == "__main__":
  main()
