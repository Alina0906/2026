from __future__ import annotations

import argparse
import csv
import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ("no", "sphere", "vort")
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
DEFAULT_DATA_ROOT = Path("dataset") / "dataset"
DEFAULT_OUTPUT_ROOT = Path("runs") / "test_iv"


@dataclass(frozen=True)
class PresetSpec:
    epochs: int
    batch_size: int
    learning_rate: float
    image_size: int
    weight_decay: float
    warmup_epochs: int
    patience: int
    dropout: float
    label_smoothing: float
    cnn_width: int
    fno_width: int
    fno_depth: int
    fno_modes: int


PRESET_SPECS: dict[str, PresetSpec] = {
    "quick": PresetSpec(
        epochs=10,
        batch_size=12,
        learning_rate=6e-4,
        image_size=96,
        weight_decay=1e-4,
        warmup_epochs=1,
        patience=2,
        dropout=0.05,
        label_smoothing=0.0,
        cnn_width=24,
        fno_width=40,
        fno_depth=3,
        fno_modes=12,
    ),
    "default": PresetSpec(
        epochs=50,
        batch_size=48,
        learning_rate=4e-4,
        image_size=128,
        weight_decay=1e-4,
        warmup_epochs=2,
        patience=4,
        dropout=0.08,
        label_smoothing=0.02,
        cnn_width=32,
        fno_width=48,
        fno_depth=4,
        fno_modes=16,
    ),
    "highres": PresetSpec(
        epochs=18,
        batch_size=12,
        learning_rate=3e-4,
        image_size=150,
        weight_decay=1e-4,
        warmup_epochs=3,
        patience=6,
        dropout=0.10,
        label_smoothing=0.03,
        cnn_width=40,
        fno_width=64,
        fno_depth=4,
        fno_modes=24,
    ),
}


@dataclass
class ExperimentConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    backbone: str = "both"
    preset: str = "default"
    run_name: str = "comparison"
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    image_size: int | None = None
    disable_early_stopping: bool = False
    tta: str = "light"
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    limit_per_class: int | None = None
    verbose: bool = True
    log_to_file: bool = False


@dataclass(frozen=True)
class ResolvedExperiment:
    data_root: Path
    run_dir: Path
    backbone: str
    preset: str
    run_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    image_size: int
    disable_early_stopping: bool
    tta: str
    seed: int
    num_workers: int
    device: str
    limit_per_class: int | None
    verbose: bool
    log_to_file: bool
    weight_decay: float
    warmup_epochs: int
    patience: int
    dropout: float
    label_smoothing: float
    cnn_width: int
    fno_width: int
    fno_depth: int
    fno_modes: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device) -> Any:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def resolve_config(config: ExperimentConfig) -> ResolvedExperiment:
    if config.preset not in PRESET_SPECS:
        raise ValueError(f"Unsupported preset: {config.preset}")
    if config.backbone not in {"cnn", "fno", "both"}:
        raise ValueError(f"Unsupported backbone: {config.backbone}")
    if config.tta not in {"none", "light", "full"}:
        raise ValueError(f"Unsupported TTA mode: {config.tta}")

    preset = PRESET_SPECS[config.preset]
    return ResolvedExperiment(
        data_root=config.data_root,
        run_dir=config.output_root / config.run_name,
        backbone=config.backbone,
        preset=config.preset,
        run_name=config.run_name,
        epochs=config.epochs if config.epochs is not None else preset.epochs,
        batch_size=config.batch_size if config.batch_size is not None else preset.batch_size,
        learning_rate=config.learning_rate if config.learning_rate is not None else preset.learning_rate,
        image_size=config.image_size if config.image_size is not None else preset.image_size,
        disable_early_stopping=config.disable_early_stopping,
        tta=config.tta,
        seed=config.seed,
        num_workers=config.num_workers,
        device=config.device,
        limit_per_class=config.limit_per_class,
        verbose=config.verbose,
        log_to_file=config.log_to_file,
        weight_decay=preset.weight_decay,
        warmup_epochs=preset.warmup_epochs,
        patience=preset.patience,
        dropout=preset.dropout,
        label_smoothing=preset.label_smoothing,
        cnn_width=preset.cnn_width,
        fno_width=preset.fno_width,
        fno_depth=preset.fno_depth,
        fno_modes=preset.fno_modes,
    )


def _collect_class_files(data_root: Path, class_name: str) -> list[Path]:
    files: list[Path] = []
    for split_name in ("train", "val"):
        files.extend(sorted((data_root / split_name / class_name).glob("*.npy")))
    return files


def _subsample_files(files: list[Path], limit_per_class: int | None, seed: int) -> list[Path]:
    if limit_per_class is None or limit_per_class >= len(files):
        return files
    subset = list(files)
    rng = random.Random(seed)
    rng.shuffle(subset)
    subset = subset[:limit_per_class]
    subset.sort()
    return subset


def build_holdout_split(
    data_root: Path,
    split_seed: int,
    limit_per_class: int | None = None,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], dict[str, Any]]:
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    combined_counts: dict[str, int] = {}

    for offset, class_name in enumerate(CLASS_NAMES):
        files = _collect_class_files(data_root, class_name)
        files = _subsample_files(files, limit_per_class=limit_per_class, seed=split_seed + offset)
        rng = random.Random(split_seed + offset)
        rng.shuffle(files)

        if len(files) <= 1:
            train_files = files
            val_files: list[Path] = []
        else:
            train_size = max(1, int(0.9 * len(files)))
            train_size = min(train_size, len(files) - 1)
            train_files = files[:train_size]
            val_files = files[train_size:]

        label = CLASS_TO_INDEX[class_name]
        combined_counts[class_name] = len(files)
        train_samples.extend((path, label) for path in train_files)
        val_samples.extend((path, label) for path in val_files)

    rng = random.Random(split_seed)
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    summary = {
        "split": "stratified_holdout_90_10",
        "combined_size": len(train_samples) + len(val_samples),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "combined_counts": combined_counts,
        "train_counts": {
            class_name: sum(label == CLASS_TO_INDEX[class_name] for _, label in train_samples)
            for class_name in CLASS_NAMES
        },
        "val_counts": {
            class_name: sum(label == CLASS_TO_INDEX[class_name] for _, label in val_samples)
            for class_name in CLASS_NAMES
        },
    }
    return train_samples, val_samples, summary


def _normalize_field(tensor: torch.Tensor) -> torch.Tensor:
    min_value = float(tensor.amin())
    max_value = float(tensor.amax())
    if max_value - min_value > 1e-6:
        tensor = (tensor - min_value) / (max_value - min_value)
    mean = tensor.mean()
    std = tensor.std().clamp_min(1e-6)
    return (tensor - mean) / std


def _augment_field(tensor: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        tensor = torch.flip(tensor, dims=(-1,))
    if random.random() < 0.5:
        tensor = torch.flip(tensor, dims=(-2,))
    rotations = random.randint(0, 3)
    if rotations:
        tensor = torch.rot90(tensor, k=rotations, dims=(-2, -1))
    if random.random() < 0.35:
        tensor = tensor * random.uniform(0.95, 1.05)
    if random.random() < 0.35:
        tensor = tensor + 0.03 * torch.randn_like(tensor)
    return tensor


class LensFieldDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        image_size: int,
        augment: bool,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        file_path, label = self.samples[index]
        image = np.load(file_path).astype(np.float32)
        tensor = torch.from_numpy(image)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        if tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        tensor = _normalize_field(tensor)
        if self.augment:
            tensor = _augment_field(tensor)
        return tensor.contiguous(), label


def build_dataloaders(
    resolved: ResolvedExperiment,
) -> tuple[DataLoader[tuple[torch.Tensor, int]], DataLoader[tuple[torch.Tensor, int]], dict[str, Any]]:
    train_samples, val_samples, summary = build_holdout_split(
        data_root=resolved.data_root,
        split_seed=resolved.seed,
        limit_per_class=resolved.limit_per_class,
    )
    train_dataset = LensFieldDataset(train_samples, image_size=resolved.image_size, augment=True)
    val_dataset = LensFieldDataset(val_samples, image_size=resolved.image_size, augment=False)

    persistent_workers = resolved.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=resolved.batch_size,
        shuffle=True,
        num_workers=resolved.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=resolved.batch_size,
        shuffle=False,
        num_workers=resolved.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, summary


def coordinate_grid(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((yy, xx), dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)


class ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ConvStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.gate = ChannelGate(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.activation = nn.GELU()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.proj(x)
        x = self.mix(x)
        x = self.gate(x)
        x = self.dropout(x)
        return self.activation(x + residual)


class FieldCNNClassifier(nn.Module):
    def __init__(self, width: int, num_classes: int = 3, dropout: float = 0.08) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            ConvStage(width, width, stride=1, dropout=dropout),
            ConvStage(width, width * 2, stride=2, dropout=dropout),
            ConvStage(width * 2, width * 3, stride=2, dropout=dropout),
            ConvStage(width * 3, width * 4, stride=2, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(width * 4, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.encoder(x)
        return self.head(x)


class FourierMixer2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_h: int, modes_w: int) -> None:
        super().__init__()
        scale = 1.0 / max(1, in_channels * out_channels)
        self.out_channels = out_channels
        self.modes_h = modes_h
        self.modes_w = modes_w
        self.weight_top = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_h, modes_w, 2))
        self.weight_bottom = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_h, modes_w, 2))

    @staticmethod
    def _multiply(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        output_dtype = x.dtype
        modes_h = min(self.modes_h, height)
        modes_w = min(self.modes_w, width // 2 + 1)

        x_fft_input = x.float() if x.dtype != torch.float32 else x
        spectrum = torch.fft.rfft2(x_fft_input, norm="ortho")
        out_spectrum = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        weight_top = torch.view_as_complex(self.weight_top[:, :, :modes_h, :modes_w, :].contiguous())
        weight_bottom = torch.view_as_complex(self.weight_bottom[:, :, :modes_h, :modes_w, :].contiguous())

        out_spectrum[:, :, :modes_h, :modes_w] = self._multiply(
            spectrum[:, :, :modes_h, :modes_w],
            weight_top,
        )
        out_spectrum[:, :, -modes_h:, :modes_w] = self._multiply(
            spectrum[:, :, -modes_h:, :modes_w],
            weight_bottom,
        )
        field = torch.fft.irfft2(out_spectrum, s=(height, width), norm="ortho")
        return field.to(dtype=output_dtype)


class OperatorBlock(nn.Module):
    def __init__(self, channels: int, modes: int, dropout: float) -> None:
        super().__init__()
        self.spectral = FourierMixer2d(channels, channels, modes_h=modes, modes_w=modes)
        self.local = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm1 = nn.GroupNorm(1, channels)
        self.feedforward = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
        )
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout2d(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.local(x)
        y = self.activation(self.norm1(y))
        x = x + self.dropout(y)

        y = self.feedforward(x)
        y = self.activation(self.norm2(y))
        return x + self.dropout(y)


class NeuralOperatorClassifier(nn.Module):
    def __init__(self, width: int, depth: int, modes: int, num_classes: int = 3, dropout: float = 0.08) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(OperatorBlock(width, modes=modes, dropout=dropout) for _ in range(depth))
        self.head = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        coords = coordinate_grid(batch_size, height, width, x.device, x.dtype)
        x = torch.cat((x, coords), dim=1)
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def make_model(backbone: str, resolved: ResolvedExperiment) -> nn.Module:
    if backbone == "cnn":
        return FieldCNNClassifier(width=resolved.cnn_width, dropout=resolved.dropout)
    if backbone == "fno":
        return NeuralOperatorClassifier(
            width=resolved.fno_width,
            depth=resolved.fno_depth,
            modes=resolved.fno_modes,
            dropout=resolved.dropout,
        )
    raise ValueError(f"Unsupported backbone: {backbone}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    if total_epochs <= 1 or warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs), eta_min=1e-6)

    warmup_epochs = min(warmup_epochs, max(0, total_epochs - 1))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.2,
        end_factor=1.0,
        total_iters=max(1, warmup_epochs),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=1e-6,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def tta_views(images: torch.Tensor, mode: str) -> list[torch.Tensor]:
    views = [images]
    if mode in {"light", "full"}:
        views.append(torch.flip(images, dims=(-1,)))
        views.append(torch.flip(images, dims=(-2,)))
    if mode == "full":
        views.append(torch.rot90(images, k=1, dims=(-2, -1)))
        views.append(torch.rot90(images, k=3, dims=(-2, -1)))
    return views


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((predictions == labels).sum().item())
        total_samples += batch_size

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
    tta_mode: str,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            views = tta_views(images, mode=tta_mode)
            probabilities_sum = None
            logits_for_loss = None
            for view in views:
                with autocast_context(device):
                    logits = model(view)
                probs = torch.softmax(logits, dim=1)
                probabilities_sum = probs if probabilities_sum is None else probabilities_sum + probs
                if logits_for_loss is None:
                    logits_for_loss = logits

            averaged_probabilities = probabilities_sum / len(views)
            loss = criterion(logits_for_loss, labels)
            predictions = averaged_probabilities.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((predictions == labels).sum().item())
            total_samples += batch_size
            all_targets.append(labels.cpu().numpy())
            all_probabilities.append(averaged_probabilities.cpu().numpy())

    y_true = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int64)
    y_prob = np.concatenate(all_probabilities) if all_probabilities else np.zeros((0, len(CLASS_NAMES)), dtype=np.float32)
    y_pred = y_prob.argmax(axis=1) if len(y_prob) else np.array([], dtype=np.int64)
    y_true_bin = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))

    per_class_auc: dict[str, float | None] = {}
    roc_curves: dict[str, dict[str, list[float]]] = {}
    valid_auc_values: list[float] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        binary_targets = y_true_bin[:, class_index] if len(y_true_bin) else np.array([], dtype=np.int64)
        class_probabilities = y_prob[:, class_index] if len(y_prob) else np.array([], dtype=np.float32)
        if binary_targets.size == 0 or np.unique(binary_targets).size < 2:
            per_class_auc[class_name] = None
            roc_curves[class_name] = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}
            continue

        fpr, tpr, _ = roc_curve(binary_targets, class_probabilities)
        auc_value = float(np.trapezoid(tpr, fpr))
        per_class_auc[class_name] = auc_value
        roc_curves[class_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        valid_auc_values.append(auc_value)

    macro_auc = float(np.mean(valid_auc_values)) if valid_auc_values else float("nan")
    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    confidence = y_prob.max(axis=1) if len(y_prob) else np.array([], dtype=np.float32)

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
        "macro_auc": macro_auc,
        "per_class_auc": per_class_auc,
        "roc_curves": roc_curves,
        "confusion_matrix": conf_mat.tolist(),
        "predicted_class_distribution": {
            class_name: int((y_pred == index).sum())
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "target_class_distribution": {
            class_name: int((y_true == index).sum())
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "confidence_mean": float(confidence.mean()) if confidence.size else float("nan"),
        "confidence_std": float(confidence.std()) if confidence.size else float("nan"),
    }


def save_json(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_log_path(resolved: ResolvedExperiment) -> Path:
    return resolved.run_dir / "train.log"


def log_run_message(resolved: ResolvedExperiment, message: str) -> None:
    if resolved.verbose:
        print(message)
    if resolved.log_to_file:
        with run_log_path(resolved).open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")


def save_history(rows: list[dict[str, float]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def create_figure(figsize: tuple[float, float]) -> Figure:
    figure = Figure(figsize=figsize)
    FigureCanvasAgg(figure)
    return figure


def plot_roc_curves(metrics: dict[str, Any], title: str, path: Path) -> None:
    fig = create_figure((7, 5))
    ax = fig.subplots()
    for class_name in CLASS_NAMES:
        curve = metrics["roc_curves"][class_name]
        auc_value = metrics["per_class_auc"][class_name]
        auc_text = "n/a" if auc_value is None else f"{auc_value:.4f}"
        ax.plot(curve["fpr"], curve["tpr"], label=f"{class_name} | AUC={auc_text}")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def plot_confusion_matrix_figure(metrics: dict[str, Any], title: str, path: Path) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    fig = create_figure((5, 4))
    ax = fig.subplots()
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def plot_training_curves(rows: list[dict[str, float]], title: str, path: Path) -> None:
    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    train_accuracy = [float(row["train_accuracy"]) for row in rows]
    val_accuracy = [float(row["val_accuracy"]) for row in rows]
    val_auc = [float(row["val_macro_auc"]) for row in rows]

    fig = create_figure((14, 4))
    axes = fig.subplots(1, 3)
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_accuracy, label="train")
    axes[1].plot(epochs, val_accuracy, label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, val_auc, color="#d35400")
    axes[2].set_title("Validation Macro AUC")
    axes[2].set_xlabel("Epoch")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "macro_auc": metrics["macro_auc"],
        "best_epoch": metrics["best_epoch"],
        "num_parameters": metrics["num_parameters"],
        "per_class_auc": metrics["per_class_auc"],
    }


def run_backbone(
    backbone: str,
    resolved: ResolvedExperiment,
    train_loader: DataLoader[tuple[torch.Tensor, int]],
    val_loader: DataLoader[tuple[torch.Tensor, int]],
) -> dict[str, Any]:
    device = torch.device(resolved.device)
    model = make_model(backbone, resolved).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=resolved.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=resolved.learning_rate,
        weight_decay=resolved.weight_decay,
    )
    scheduler = build_scheduler(optimizer, total_epochs=resolved.epochs, warmup_epochs=resolved.warmup_epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    model_dir = resolved.run_dir / backbone
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "best_model.pt"

    history: list[dict[str, float]] = []
    best_epoch = 0
    best_macro_auc = float("-inf")
    stale_epochs = 0

    for epoch in range(1, resolved.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            tta_mode="none",
        )
        scheduler.step()

        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_auc": float(val_metrics["macro_auc"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        improved = val_metrics["macro_auc"] > best_macro_auc
        if improved:
            best_macro_auc = float(val_metrics["macro_auc"])
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            stale_epochs += 1

        log_run_message(
            resolved,
            f"[{backbone}] epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_auc={val_metrics['macro_auc']:.4f}"
            + (" *" if improved else "")
        )

        if (not resolved.disable_early_stopping) and stale_epochs >= resolved.patience:
            log_run_message(resolved, f"[{backbone}] early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    final_metrics = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        tta_mode=resolved.tta,
    )
    final_metrics["best_epoch"] = best_epoch
    final_metrics["num_parameters"] = int(sum(parameter.numel() for parameter in model.parameters()))
    final_metrics["tta_mode"] = resolved.tta

    save_history(history, model_dir / "history.csv")
    save_json(final_metrics, model_dir / "metrics.json")
    plot_roc_curves(final_metrics, title=f"{backbone.upper()} ROC analysis", path=model_dir / "roc_curves.png")
    plot_confusion_matrix_figure(
        final_metrics,
        title=f"{backbone.upper()} confusion analysis",
        path=model_dir / "confusion_matrix.png",
    )
    plot_training_curves(history, title=f"{backbone.upper()} learning dynamics", path=model_dir / "training_curves.png")
    return final_metrics


def _resolved_to_dict(resolved: ResolvedExperiment) -> dict[str, Any]:
    return {
        "data_root": str(resolved.data_root),
        "run_dir": str(resolved.run_dir),
        "backbone": resolved.backbone,
        "preset": resolved.preset,
        "run_name": resolved.run_name,
        "epochs": resolved.epochs,
        "batch_size": resolved.batch_size,
        "learning_rate": resolved.learning_rate,
        "image_size": resolved.image_size,
        "disable_early_stopping": resolved.disable_early_stopping,
        "tta": resolved.tta,
        "seed": resolved.seed,
        "num_workers": resolved.num_workers,
        "device": resolved.device,
        "limit_per_class": resolved.limit_per_class,
        "verbose": resolved.verbose,
        "log_to_file": resolved.log_to_file,
        "weight_decay": resolved.weight_decay,
        "warmup_epochs": resolved.warmup_epochs,
        "patience": resolved.patience,
        "dropout": resolved.dropout,
        "label_smoothing": resolved.label_smoothing,
        "cnn_width": resolved.cnn_width,
        "fno_width": resolved.fno_width,
        "fno_depth": resolved.fno_depth,
        "fno_modes": resolved.fno_modes,
    }


def run_suite(config: ExperimentConfig) -> dict[str, Any]:
    resolved = resolve_config(config)
    set_seed(resolved.seed)

    resolved.run_dir.mkdir(parents=True, exist_ok=True)
    if resolved.log_to_file:
        run_log_path(resolved).write_text("", encoding="utf-8")
    save_json(_resolved_to_dict(resolved), resolved.run_dir / "config.json")

    train_loader, val_loader, split_summary = build_dataloaders(resolved)
    save_json(split_summary, resolved.run_dir / "split_summary.json")

    backbones = ["cnn", "fno"] if resolved.backbone == "both" else [resolved.backbone]
    results: dict[str, dict[str, Any]] = {}
    for backbone in backbones:
        results[backbone] = run_backbone(
            backbone=backbone,
            resolved=resolved,
            train_loader=train_loader,
            val_loader=val_loader,
        )

    suite_summary: dict[str, Any] = {
        "run_dir": str(resolved.run_dir),
        "preset": resolved.preset,
        "backbones": backbones,
        "split_summary": split_summary,
        "results": {name: summarize_metrics(metrics) for name, metrics in results.items()},
    }
    if len(backbones) > 1:
        best_backbone = max(backbones, key=lambda name: float(results[name]["macro_auc"]))
        suite_summary["best_backbone"] = best_backbone
        suite_summary["macro_auc_delta_fno_vs_cnn"] = float(results["fno"]["macro_auc"] - results["cnn"]["macro_auc"])
        suite_summary["accuracy_delta_fno_vs_cnn"] = float(results["fno"]["accuracy"] - results["cnn"]["accuracy"])

    save_json(suite_summary, resolved.run_dir / "suite_summary.json")
    return suite_summary


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    if config.backbone == "both":
        raise ValueError("run_experiment expects a single backbone; use run_suite for dual-backbone runs.")
    return run_suite(config)


def load_run_artifacts(run_dir: Path) -> dict[str, Any]:
    artifacts = {
        "config": {},
        "split_summary": {},
        "suite_summary": {},
        "models": {},
    }

    config_path = run_dir / "config.json"
    split_path = run_dir / "split_summary.json"
    suite_path = run_dir / "suite_summary.json"
    if config_path.exists():
        artifacts["config"] = json.loads(config_path.read_text(encoding="utf-8"))
    if split_path.exists():
        artifacts["split_summary"] = json.loads(split_path.read_text(encoding="utf-8"))
    if suite_path.exists():
        artifacts["suite_summary"] = json.loads(suite_path.read_text(encoding="utf-8"))

    for backbone in ("cnn", "fno"):
        model_dir = run_dir / backbone
        if not model_dir.exists():
            continue
        metrics_path = model_dir / "metrics.json"
        artifacts["models"][backbone] = {
            "metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {},
            "history": load_history(model_dir / "history.csv"),
            "paths": {
                "roc_curves": model_dir / "roc_curves.png",
                "training_curves": model_dir / "training_curves.png",
                "confusion_matrix": model_dir / "confusion_matrix.png",
                "weights": model_dir / "best_model.pt",
            },
        }
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Common Test I unified experiment runner")
    parser.add_argument("--backbone", choices=["cnn", "fno", "both"], default="both")
    parser.add_argument("--preset", choices=["quick", "default", "highres"], default="default")
    parser.add_argument("--run-name", type=str, default="comparison")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--tta", choices=["none", "light", "full"], default="light")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-per-class", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        backbone=args.backbone,
        preset=args.preset,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        disable_early_stopping=args.disable_early_stopping,
        tta=args.tta,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        limit_per_class=args.limit_per_class,
    )
    summary = run_suite(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
