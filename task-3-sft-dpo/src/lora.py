"""LoRA injection implemented with basic PyTorch modules."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / r
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: Tensor) -> Tensor:
        adapter = self.lora_b(self.lora_a(self.lora_dropout(inputs)))
        return self.base(inputs) + adapter * self.scaling

    def merged_linear(self) -> nn.Linear:
        with torch.no_grad():
            delta = torch.matmul(self.lora_b.weight, self.lora_a.weight) * self.scaling
            self.base.weight.add_(delta.to(self.base.weight.dtype))
        return self.base


def inject_lora(
    model: nn.Module,
    target_modules: list[str],
    r: int,
    alpha: float,
    dropout: float = 0.0,
) -> nn.Module:
    """Freeze ``model`` and replace matching linear layers with LoRA wrappers."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    targets = set(target_modules)
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    for _, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if child_name in targets and isinstance(child, nn.Linear):
                replacements.append((parent, child_name, child))
    if not replacements:
        raise ValueError(f"no nn.Linear modules matched {sorted(targets)}")
    for parent, name, layer in replacements:
        wrapper = LoRALinear(layer, r=r, alpha=alpha, dropout=dropout)
        wrapper.to(device=layer.weight.device, dtype=layer.weight.dtype)
        setattr(parent, name, wrapper)
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    replacements: list[tuple[nn.Module, str, LoRALinear]] = []
    for _, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, LoRALinear):
                replacements.append((parent, child_name, child))
    for parent, name, wrapper in replacements:
        setattr(parent, name, wrapper.merged_linear())
    return model


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if ".lora_a." in name or ".lora_b." in name
    }


def save_lora(
    model: nn.Module,
    output_dir: str | Path,
    *,
    target_modules: list[str],
    r: int,
    alpha: float,
    **metadata,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), output_dir / "adapter.pt")
    (output_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "target_modules": target_modules,
                "r": r,
                "alpha": alpha,
                **metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_lora(model: nn.Module, adapter_dir: str | Path) -> nn.Module:
    adapter_dir = Path(adapter_dir)
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if not any(isinstance(module, LoRALinear) for module in model.modules()):
        inject_lora(
            model,
            target_modules=config["target_modules"],
            r=int(config["r"]),
            alpha=float(config["alpha"]),
        )
    state = torch.load(adapter_dir / "adapter.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected adapter keys: {unexpected}")
    return model
