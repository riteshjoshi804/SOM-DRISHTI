"""Loading and caching of pretrained model weights.

Models are initialised once per process and shared between requests. Weights live
in ``models/`` under the project root (never in Git) and are fetched only from the
upstream locations recorded in ``WEIGHTS``. If they cannot be obtained the matcher
reports UNAVAILABLE with the reason; nothing falls back to random weights.

PyTorch and its companions are imported lazily, so the classical SIFT pipeline, the
API and the tests of everything else keep working without them.
"""
from __future__ import annotations

import importlib.util
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from app import config
from app.matching.base import WeightInfo


class ModelUnavailable(RuntimeError):
    """A matcher cannot run: missing dependency, or weights that cannot be obtained."""


@dataclass(frozen=True)
class WeightSpec:
    name: str
    relative_path: str  # below MODELS_ROOT
    source_url: str
    license: str


WEIGHTS: dict[str, WeightSpec] = {
    # Licences checked against the upstream files on 2026-10-30.
    "superpoint": WeightSpec(
        name="SuperPoint (Magic Leap, Hugging Face port)",
        relative_path="huggingface/models--magic-leap-community--superpoint",
        source_url="https://huggingface.co/magic-leap-community/superpoint (Magic Leap weights)",
        license="Magic Leap licence: academic or non-profit organization noncommercial research use only",
    ),
    "lightglue": WeightSpec(
        name="LightGlue for SuperPoint (ETH CVG, via Kornia)",
        relative_path="torch/checkpoints/superpoint_lightglue_v0-1_arxiv-pth",
        source_url="https://huggingface.co/kornia/lightglue, falling back to the cvg/LightGlue v0.1_arxiv release",
        license="Apache-2.0",
    ),
    "loftr": WeightSpec(
        name="LoFTR outdoor (MegaDepth, via Kornia)",
        relative_path="torch/checkpoints/loftr_outdoor.ckpt",
        source_url="https://huggingface.co/kornia/loftr, falling back to cmp.felk.cvut.cz/~mishkdmy/models",
        license="Apache-2.0 (zju3dv/LoFTR)",
    ),
}

_lock = threading.RLock()
_loaded: dict[str, object] = {}
_device: str | None = None


def dependency_missing(*modules: str) -> str | None:
    """Name of the first module that is not installed, without importing anything heavy."""
    for module in modules:
        if importlib.util.find_spec(module) is None:
            return module
    return None


def weight_path(key: str) -> Path:
    return config.MODELS_ROOT / WEIGHTS[key].relative_path


def _size_of(path: Path) -> int | None:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        # Hugging Face snapshots hold pointers; only blobs carry bytes (no symlinks on Windows by default).
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink()) or None
    return None


def weights_present(key: str) -> bool:
    path = weight_path(key)
    if path.is_dir():
        return (path / "snapshots").is_dir() and any(path.glob("snapshots/*/model.safetensors"))
    return path.is_file() and path.stat().st_size > 0


def weight_info(key: str) -> WeightInfo:
    spec, path = WEIGHTS[key], weight_path(key)
    present = weights_present(key)
    return WeightInfo(
        name=spec.name,
        source_url=spec.source_url,
        license=spec.license,
        present=present,
        path=str(path) if present else None,
        size_bytes=_size_of(path) if present else None,
    )


def _prepare_environment() -> None:
    """Keep all downloads under models/ and silence notices that do not apply here."""
    import torch

    root = config.MODELS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(root / "torch"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def resolve_device(preference: str = "auto") -> str:
    """'cuda' when requested or available, otherwise 'cpu'. Cached per process for 'auto'."""
    global _device
    import torch

    if preference != "auto":
        if preference.startswith("cuda") and not torch.cuda.is_available():
            raise ModelUnavailable(f"device '{preference}' requested but CUDA is not available")
        return preference
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def device_label(device: str) -> str:
    """Human-readable device, for reports."""
    import torch

    if device.startswith("cuda"):
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return f"cpu ({torch.get_num_threads()} threads)"


def _cached(key: str, build):
    with _lock:
        if key not in _loaded:
            _loaded[key] = build()
        return _loaded[key]


def load_superpoint(device: str, *, allow_download: bool = True):
    def build():
        missing = dependency_missing("torch", "transformers")
        if missing:
            raise ModelUnavailable(f"Python package '{missing}' is not installed")
        _prepare_environment()
        if not weights_present("superpoint") and not allow_download:
            raise ModelUnavailable("SuperPoint weights are not downloaded")
        from transformers import SuperPointForKeypointDetection

        cache_dir = str(config.MODELS_ROOT / "huggingface")
        try:
            model = SuperPointForKeypointDetection.from_pretrained(
                "magic-leap-community/superpoint", cache_dir=cache_dir, local_files_only=weights_present("superpoint")
            )
        except Exception as exc:  # network, disk or format problems all mean: not operational
            raise ModelUnavailable(f"SuperPoint weights could not be loaded: {exc}") from exc
        return model.eval().to(device)

    return _cached(f"superpoint:{device}", build)


def load_lightglue(device: str, *, allow_download: bool = True):
    def build():
        missing = dependency_missing("torch", "kornia")
        if missing:
            raise ModelUnavailable(f"Python package '{missing}' is not installed")
        _prepare_environment()
        if not weights_present("lightglue") and not allow_download:
            raise ModelUnavailable("LightGlue weights are not downloaded")
        import kornia.feature as KF

        try:
            # Thresholds are applied per call (see superpoint_lightglue.py): the model keeps
            # everything above a floor, so one instance serves any configured threshold.
            model = KF.LightGlue("superpoint", filter_threshold=0.01)
        except Exception as exc:
            raise ModelUnavailable(f"LightGlue weights could not be loaded: {exc}") from exc
        return model.eval().to(device)

    return _cached(f"lightglue:{device}", build)


def load_loftr(device: str, *, allow_download: bool = True):
    def build():
        missing = dependency_missing("torch", "kornia")
        if missing:
            raise ModelUnavailable(f"Python package '{missing}' is not installed")
        _prepare_environment()
        if not weights_present("loftr") and not allow_download:
            raise ModelUnavailable("LoFTR weights are not downloaded")
        import kornia.feature as KF

        try:
            model = KF.LoFTR(pretrained="outdoor")
        except Exception as exc:
            raise ModelUnavailable(f"LoFTR weights could not be loaded: {exc}") from exc
        return model.eval().to(device)

    return _cached(f"loftr:{device}", build)


def unload_all() -> None:
    """Release every cached model (used by tests and long-running batch scripts)."""
    with _lock:
        _loaded.clear()
