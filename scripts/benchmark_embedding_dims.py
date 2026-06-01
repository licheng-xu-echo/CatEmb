#!/usr/bin/env python3
"""Compute 2D-3D alignment for CatEmb checkpoints with different dimensions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from catemb.data import get_idx_split, process_single_mol  # noqa: E402
from catemb.model import CL2D3DMol  # noqa: E402

DEFAULT_EXTERNAL_MODEL_ROOT = Path(
    "/inspire/hdd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/"
    "8359-xulicheng/CatEmb/save_model"
)


@dataclass
class ModelBundle:
    dim: int
    model_dir: Path
    full_params: dict[str, Any]
    checkpoint: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--external-model-root", type=Path, default=DEFAULT_EXTERNAL_MODEL_ROOT)
    parser.add_argument("--model-dir", dest="model_dirs", type=Path, nargs="*", default=None)
    parser.add_argument(
        "--repr-data-path",
        type=Path,
        default=REPO_ROOT / "dataset" / "processed" / "catcompdb.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmark_results" / "embedding_dims",
    )
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def discover_model_dirs(repo_root: Path, external_model_root: Path, explicit_model_dirs: list[Path] | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_model_dirs:
        candidates.extend(explicit_model_dirs)
    else:
        local_dim32 = repo_root / "catemb" / "model_path" / "dim32LN"
        if local_dim32.exists():
            candidates.append(local_dim32)
        if external_model_root.exists():
            candidates.extend(path for path in sorted(external_model_root.iterdir()) if path.is_dir())

    dedup: dict[int, Path] = {}
    for model_dir in candidates:
        params_path = model_dir / "full_params.npy"
        ckpt_path = model_dir / "best_model.pt"
        if not (params_path.exists() and ckpt_path.exists()):
            continue
        params = np.load(params_path, allow_pickle=True).item()
        dim = int(params["param_2d"]["dest_dim"])
        dedup[dim] = model_dir
    return [dedup[dim] for dim in sorted(dedup)]


def load_model_bundle(model_dir: Path) -> ModelBundle:
    full_params = np.load(model_dir / "full_params.npy", allow_pickle=True).item()
    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=False)
    return ModelBundle(
        dim=int(full_params["param_2d"]["dest_dim"]),
        model_dir=model_dir,
        full_params=full_params,
        checkpoint=checkpoint,
    )


def build_model(bundle: ModelBundle, device: str) -> CL2D3DMol:
    model = CL2D3DMol(
        param_2d=bundle.full_params["param_2d"],
        param_3d=bundle.full_params["param_3d"],
        reduce=bundle.full_params["param_cl"]["reduce"],
    )
    model.load_state_dict(bundle.checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norm > 0.0, norm, 1.0)


def compute_alignment(z2d: np.ndarray, z3d: np.ndarray) -> float:
    z2d = normalize_rows(z2d)
    z3d = normalize_rows(z3d)
    return float(np.square(z2d - z3d).sum(axis=1).mean())


def load_dataloader(path: Path, seed: int, max_samples: int, batch_size: int) -> tuple[DataLoader, int]:
    raw = np.load(path, allow_pickle=True)
    split = get_idx_split(
        data_size=len(raw),
        train_size=int(len(raw) * 0.9),
        valid_size=int(len(raw) * 0.1),
        seed=seed,
    )
    valid_indices = split["valid"].tolist()[:max_samples]
    data_list = [
        data
        for idx in valid_indices
        if (data := process_single_mol((idx, raw[idx], seed, True, True))) is not None
    ]
    return DataLoader(data_list, batch_size=batch_size, shuffle=False), len(data_list)


def evaluate_alignment(bundle: ModelBundle, args: argparse.Namespace) -> dict[str, Any]:
    loader, sample_size = load_dataloader(args.repr_data_path, args.seed, args.max_samples, args.batch_size)
    model = build_model(bundle, args.device)
    z2d_chunks: list[np.ndarray] = []
    z3d_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(args.device)
            z2d, z3d, _ = model(batch)
            z2d_chunks.append(z2d.detach().cpu().numpy())
            z3d_chunks.append(z3d.detach().cpu().numpy())
    if not z2d_chunks:
        raise RuntimeError(f"No valid molecules were loaded from {args.repr_data_path}")
    return {
        "dim": bundle.dim,
        "alignment_2d3d": compute_alignment(np.concatenate(z2d_chunks), np.concatenate(z3d_chunks)),
        "repr_sample_size": sample_size,
        "model_dir": str(bundle.model_dir),
    }


def main() -> None:
    args = parse_args()
    model_dirs = discover_model_dirs(args.repo_root, args.external_model_root, args.model_dirs)
    if not model_dirs:
        raise SystemExit("No valid model directories were found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_dir in model_dirs:
        bundle = load_model_bundle(model_dir)
        print(f"[alignment] dim={bundle.dim} model_dir={bundle.model_dir}")
        rows.append(evaluate_alignment(bundle, args))

    summary = pd.DataFrame(rows).sort_values("dim")
    summary_path = args.output_dir / "summary.csv"
    details_path = args.output_dir / "details.json"
    summary.to_csv(summary_path, index=False)
    details_path.write_text(json.dumps({"models": rows}, indent=2))
    print(f"[done] summary -> {summary_path}")
    print(f"[done] details -> {details_path}")


if __name__ == "__main__":
    main()
