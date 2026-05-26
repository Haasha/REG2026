"""
core.py — Platform I/O utilities for the REG2026 submission container.

This file is not meant to be edited. It handles all fixed platform contracts:
  - canonical path constants
  - interface detection from /input/inputs.json
  - file I/O helpers (JSON, JPG)
  - GPU diagnostics

Import what you need from inference.py; do not modify this file.
"""

import json
from pathlib import Path

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Fixed platform paths — do not change these
# ---------------------------------------------------------------------------

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")


# ---------------------------------------------------------------------------
# Interface detection
# ---------------------------------------------------------------------------

def get_interface_key() -> tuple:
    """
    Read /input/inputs.json (injected by the platform) and return a sorted
    tuple of socket slugs. Used in inference.py to dispatch to the correct
    handler without any manual branching.
    """
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    slugs = [entry["socket"]["slug"] for entry in inputs]
    return tuple(sorted(slugs))


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def load_json_file(*, location: Path):
    with open(location) as f:
        return json.loads(f.read())


def write_json_file(*, location: Path, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


# ---------------------------------------------------------------------------
# Image loaders
# ---------------------------------------------------------------------------

def load_jpg_image(*, location: Path) -> Image.Image:
    """Load a JPG image as a PIL Image in RGB mode."""
    return Image.open(location).convert("RGB")


# ---------------------------------------------------------------------------
# GPU diagnostics
# ---------------------------------------------------------------------------

def show_torch_cuda_info():
    print("=+=" * 10)
    print("Torch CUDA available:", (available := torch.cuda.is_available()))
    if available:
        print(f"  devices          : {torch.cuda.device_count()}")
        current = torch.cuda.current_device()
        print(f"  current device   : {current}")
        print(f"  device properties: {torch.cuda.get_device_properties(current)}")
    print("=+=" * 10)
