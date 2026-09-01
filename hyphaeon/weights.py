"""
hyphaeon/weights.py
------------------
Handles discovery, download, and caching of HyphAeon model weights from Hugging Face.

Weights are hosted at https://huggingface.co/datamonkey/hyphaeon and that repo is
the source of truth. On first use, weights are downloaded and cached locally
in the Hugging Face cache directory (~/.cache/huggingface/ by default).
Subsequent runs use the cached copy.

The repo may contain multiple model variants (e.g. general, viral). Available
variants are enumerated dynamically from the HF Hub API so new variants appear
without a package update.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, List

# NumPy 1.x / 2.x unpickling compatibility bridge
try:
    import numpy as np
    if not hasattr(np, '_core') and hasattr(np, 'core'):
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
except Exception:
    pass

from huggingface_hub import list_repo_files, hf_hub_download

HF_REPO_ID = "datamonkey/hyphaeon"
DEFAULT_VARIANT = "general"
DEFAULT_CONFIG_FILENAME = "config.json"

# Local cache directory for downloaded weights
CACHE_DIR = Path(os.environ.get("HYPHAEON_CACHE", str(Path.home() / ".cache" / "hyphaeon")))


def list_available_variants() -> List[Dict[str, str]]:
    """
    Enumerate available model variants from the HF repo by listing .safetensors files.

    Returns a list of dicts with keys: variant, filename, description.
    The 'general' variant (model.safetensors) is listed first.
    """
    files = list_repo_files(HF_REPO_ID)
    variants = []

    for f in files:
        if f == "model.safetensors":
            variants.append({
                "variant": "general",
                "filename": f,
                "description": "General model (trained on diverse alignments)",
            })
        elif f.startswith("model.") and f.endswith(".safetensors"):
            # e.g. model.viral.safetensors -> variant "viral"
            variant_name = f[len("model."):-len(".safetensors")]
            variants.append({
                "variant": variant_name,
                "filename": f,
                "description": f"{variant_name} variant",
            })

    # General first, then alphabetical
    variants.sort(key=lambda v: (v["variant"] != "general", v["variant"]))
    return variants


def get_variant_filename(variant: str) -> str:
    """Map a variant name to its HF filename."""
    if variant == "general":
        return "model.safetensors"
    return f"model.{variant}.safetensors"


def resolve_weights_path(
    weights: Optional[str] = None,
    variant: Optional[str] = None,
) -> str:
    """
    Resolve the path to a weights file, downloading from HF if needed.

    Resolution order:
    1. If `weights` is an explicit path to an existing file, use it directly.
    2. If `variant` is specified (or default), check the HF cache.
    3. If not cached, download from HF and cache locally.

    Returns the local path to the weights file.
    """
    # 1. Explicit path takes precedence
    if weights:
        if os.path.exists(weights):
            return weights
        pkg_root_weights = Path(__file__).resolve().parent.parent / weights
        if pkg_root_weights.exists():
            return str(pkg_root_weights)

    # 2. Determine which variant to use
    v = variant or DEFAULT_VARIANT
    filename = get_variant_filename(v)

    # 3. Check HF cache (from prior hf_hub_download calls)
    try:
        from huggingface_hub import try_to_load_from_cache
        hf_cache_path = try_to_load_from_cache(
            repo_id=HF_REPO_ID, filename=filename,
            cache_dir=str(CACHE_DIR.parent / "huggingface"),
        )
        if hf_cache_path is not None and os.path.exists(hf_cache_path):
            return hf_cache_path
    except Exception:
        pass

    # 4. Download from HF
    print(f"[*] Downloading HyphAeon weights ({v} variant) from Hugging Face...")
    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            cache_dir=str(CACHE_DIR.parent / "huggingface"),
        )
        print(f"[✓] Weights cached at: {downloaded}")
        return downloaded
    except Exception as e:
        raise RuntimeError(
            f"Could not download weights from Hugging Face ({e}). "
            f"Set HF_TOKEN env var (see .env.example) or specify --weights /path/to/checkpoint."
        ) from e


def load_model_config(variant: Optional[str] = None) -> Dict:
    """
    Load model architecture config from the HF repo's config.json.

    All variants share the same architecture (config.json at repo root).
    If a variant-specific config exists (model.{variant}.config.json), it
    will be used instead.
    """
    v = variant or DEFAULT_VARIANT

    # Check for variant-specific config first
    variant_config = f"model.{v}.config.json"
    files = list_repo_files(HF_REPO_ID)
    if variant_config in files:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=variant_config,
                               cache_dir=str(CACHE_DIR.parent / "huggingface"))
        with open(path) as f:
            return json.load(f)

    # Fall back to shared config.json
    if DEFAULT_CONFIG_FILENAME in files:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=DEFAULT_CONFIG_FILENAME,
                               cache_dir=str(CACHE_DIR.parent / "huggingface"))
        with open(path) as f:
            return json.load(f)

    # No config available — return empty dict (CLI will use defaults)
    return {}


def load_weights(
    weights: Optional[str] = None,
    variant: Optional[str] = None,
    map_location="cpu",
):
    """
    Load model weights as a state_dict.

    Supports both .pt and .safetensors formats.
    For .pt files, extracts the model_state_dict.
    For .safetensors, loads directly.
    """
    import torch

    path = resolve_weights_path(weights=weights, variant=variant)

    # .safetensors format
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        raw_dict = load_file(path, device=str(map_location))
        # If saved as a unified suite with backbone. prefix, map keys for standalone PhyloAxialTransformer
        mapped_dict = {}
        has_prefixed = any(k.startswith("backbone.") for k in raw_dict.keys())
        if has_prefixed:
            for k, v in raw_dict.items():
                if k.startswith("backbone."):
                    mapped_dict[k.replace("backbone.", "")] = v
                elif k.startswith("head_meme."):
                    mapped_dict[k.replace("head_meme.", "lrt_ordinal_head.")] = v
                else:
                    mapped_dict[k] = v
            return mapped_dict
        return raw_dict

    # .pt format (legacy or explicit path)
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


# Default architecture parameters, used when no config is available.
_DEFAULT_ARCH = {"embed_dim": 384, "num_layers": 6, "num_heads": 12, "window_size": 1}


def load_arch_config(
    weights: Optional[str] = None,
    variant: Optional[str] = None,
) -> dict:
    """
    Determine architecture hyperparameters (embed_dim, num_layers, num_heads, window_size).

    For .pt files: reads from 'args' dict in checkpoint.
    For .safetensors (HF): reads config.json or variant-specific config from HF.
    Falls back to default parameters if not found.
    """
    path = resolve_weights_path(weights=weights, variant=variant)

    if path.endswith(".pt"):
        import torch
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        a = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
        if not a and isinstance(ckpt, dict):
            a = ckpt
        return {
            "embed_dim": a.get("embed_dim", _DEFAULT_ARCH["embed_dim"]),
            "num_layers": a.get("num_layers", a.get("layers", _DEFAULT_ARCH["num_layers"])),
            "num_heads": a.get("num_heads", a.get("heads", _DEFAULT_ARCH["num_heads"])),
            "window_size": a.get("window_size", _DEFAULT_ARCH["window_size"]),
        }

    # .safetensors: fetch config from HF
    try:
        cfg = load_model_config(variant=variant)
        return {
            "embed_dim": cfg.get("embed_dim", _DEFAULT_ARCH["embed_dim"]),
            "num_layers": cfg.get("num_layers", cfg.get("layers", _DEFAULT_ARCH["num_layers"])),
            "num_heads": cfg.get("num_heads", cfg.get("heads", _DEFAULT_ARCH["num_heads"])),
            "window_size": cfg.get("window_size", _DEFAULT_ARCH["window_size"]),
        }
    except Exception:
        return dict(_DEFAULT_ARCH)
