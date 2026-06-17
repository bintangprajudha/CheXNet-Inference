from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models


def _state_dict_from_checkpoint(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(k, str) for k in checkpoint.keys()):
            return checkpoint
    raise ValueError("Checkpoint does not contain a recognizable PyTorch state_dict")


def _normalize_key(key: str) -> str:
    key = key.removeprefix("module.")
    key = key.removeprefix("densenet121.")
    key = key.replace(".norm.1.", ".norm1.")
    key = key.replace(".norm.2.", ".norm2.")
    key = key.replace(".conv.1.", ".conv1.")
    key = key.replace(".conv.2.", ".conv2.")
    key = key.replace("classifier.0.", "classifier.")
    return key


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    return {_normalize_key(key): value for key, value in state_dict.items()}


def load_chexnet_checkpoint(model: nn.Module, checkpoint_path: str | Path, skip_classifier: bool = True) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _normalize_state_dict_keys(_state_dict_from_checkpoint(checkpoint))
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if skip_classifier and key.startswith("classifier."):
            if key not in model_state or tuple(model_state[key].shape) != tuple(value.shape):
                skipped.append(key)
                continue
        filtered[key] = value
    result = model.load_state_dict(filtered, strict=False)
    if skipped:
        print(f"Skipped classifier keys due to shape mismatch: {skipped}")
    print(f"Missing keys: {list(result.missing_keys)}")
    print(f"Unexpected keys: {list(result.unexpected_keys)}")


def build_chexnet(num_classes: int, checkpoint_path: str | Path | None = None, freeze_backbone: bool = False) -> nn.Module:
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    if checkpoint_path:
        load_chexnet_checkpoint(model, checkpoint_path, skip_classifier=True)
    if freeze_backbone:
        set_stage_a_trainable(model)
    return model


def set_stage_a_trainable(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def set_stage_b_trainable(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for module in [model.features.denseblock4, model.features.norm5, model.classifier]:
        for param in module.parameters():
            param.requires_grad = True
