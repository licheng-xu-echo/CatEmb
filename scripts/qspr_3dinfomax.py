#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from oos_splits import thiol_split as build_thiol_oos_split

DEFAULT_INFOMAX_ROOT = Path(__file__).resolve().parents[2] / "3DInfomax"
DEFAULT_CONFIG = Path("configs_clean/fingerprint_inference.yml")
MODEL_TYPES = ["ExtraTrees", "Ridge", "KernelRidge", "SVR"]
OOS_SPLITS = ["substrate_oos", "catalyst_oos", "substrate_catalyst_oos"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pure 3DInfomax descriptors on random and OOS QSPR tasks.")
    p.add_argument("--dataset", choices=["aryl", "thiol", "both"], default="both")
    p.add_argument("--mode", choices=["random", "oos", "both"], default="both")
    p.add_argument("--infomax-root", type=Path, default=DEFAULT_INFOMAX_ROOT)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--model-types", nargs="+", default=MODEL_TYPES, choices=MODEL_TYPES)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--use-cache", dest="use_cache", action="store_true")
    p.add_argument("--no-cache", dest="use_cache", action="store_false")
    p.set_defaults(use_cache=True)
    p.add_argument("--cache-path", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def resolve_under_root(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open() as f:
        return yaml.safe_load(f) or {}


def prepare_3dinfomax_args(infomax_root: Path, config_path: Path, checkpoint_path: Optional[Path]) -> SimpleNamespace:
    config_path = resolve_under_root(config_path, infomax_root)
    config_dict = load_yaml(config_path)
    if checkpoint_path is None:
        checkpoint_path = resolve_under_root(Path(config_dict["checkpoint"]), infomax_root)
    else:
        checkpoint_path = resolve_under_root(checkpoint_path, infomax_root)
    checkpoint_dict = load_yaml(checkpoint_path.parent / "train_arguments.yaml")
    args_dict = dict(config_dict)
    for key, value in checkpoint_dict.items():
        if key not in args_dict:
            args_dict[key] = value
    args_dict["config"] = str(config_path)
    args_dict["checkpoint"] = str(checkpoint_path)
    return SimpleNamespace(**args_dict)


def import_3dinfomax(infomax_root: Path):
    infomax_root = infomax_root.resolve()
    if str(infomax_root) not in sys.path:
        sys.path.insert(0, str(infomax_root))
    from datasets.inference_dataset import InferenceDataset
    from inference import load_model

    return InferenceDataset, load_model


def build_3dinfomax_repr_map(
    smiles_list: List[str],
    infomax_root: Path,
    config_path: Path,
    checkpoint_path: Optional[Path],
    cache_path: Path,
    batch_size: int,
    device_name: str,
    use_cache: bool,
) -> Dict[str, np.ndarray]:
    if use_cache and cache_path.exists():
        return np.load(cache_path, allow_pickle=True).item()

    import dgl
    import torch
    from torch.utils.data import DataLoader

    InferenceDataset, load_model = import_3dinfomax(infomax_root)
    model_args = prepare_3dinfomax_args(infomax_root, config_path, checkpoint_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    unique_smiles = sorted(set(smiles_list))

    with tempfile.NamedTemporaryFile("w", suffix=".smi", delete=False) as tmp_file:
        tmp_file.write("\n".join(unique_smiles))
        tmp_file.write("\n")
        tmp_path = tmp_file.name
    try:
        dataset = InferenceDataset(smiles_txt_path=tmp_path)
    finally:
        os.unlink(tmp_path)

    model = load_model(model_args, device=device)
    checkpoint = torch.load(model_args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: dgl.batch(batch))
    chunks = []
    with torch.no_grad():
        for batch in dataloader:
            chunks.append(model(batch.to(device)).detach().cpu().numpy())

    arr = np.concatenate(chunks, axis=0)
    repr_map = {smiles: arr[i] for i, smiles in enumerate(unique_smiles)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, repr_map)
    return repr_map


def feature_matrix(repr_map: Dict[str, np.ndarray], smiles_groups: List[List[str]]) -> np.ndarray:
    return np.concatenate([np.stack([repr_map[s] for s in group], axis=0) for group in smiles_groups], axis=1)


def make_regressor(model_type: str, n_estimators: int, seed: int):
    if model_type == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    if model_type == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if model_type == "KernelRidge":
        return make_pipeline(StandardScaler(), KernelRidge(alpha=1.0, kernel="rbf"))
    if model_type == "SVR":
        return make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.1))
    raise ValueError("Unsupported model type: %s" % model_type)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }


def run_random(x: np.ndarray, y: np.ndarray, test_size: float, seeds: List[int], model_type: str, n_estimators: int, random_seed: int) -> List[Dict[str, float]]:
    rows = []
    for seed in seeds:
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, random_state=seed)
        pred = make_regressor(model_type, n_estimators, random_seed).fit(train_x, train_y).predict(test_x)
        row = {"seed": int(seed)}
        row.update(regression_metrics(test_y, pred))
        rows.append(row)
    return rows


def run_oos(x: np.ndarray, y: np.ndarray, split: Dict[str, List[int]], model_type: str, n_estimators: int, random_seed: int) -> List[Dict[str, object]]:
    train_idx = np.asarray(split["train"], dtype=int)
    model = make_regressor(model_type, n_estimators, random_seed).fit(x[train_idx], y[train_idx])
    rows = []
    for split_name in OOS_SPLITS:
        idx = np.asarray(split[split_name], dtype=int)
        row = {"split": split_name, "n_samples": int(len(idx))}
        row.update(regression_metrics(y[idx], model.predict(x[idx])))
        rows.append(row)
    return rows


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    df = pd.DataFrame(rows)
    return {
        "r2_mean": float(df.r2.mean()),
        "r2_std": float(df.r2.std(ddof=1)),
        "mae_mean": float(df.mae.mean()),
        "mae_std": float(df.mae.std(ddof=1)),
        "rmse_mean": float(df.rmse.mean()),
        "rmse_std": float(df.rmse.std(ddof=1)),
    }


def literal_constants(script_path: Path, names) -> Dict[str, object]:
    vals = {}
    for node in ast.parse(script_path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", "") in names:
            vals[node.targets[0].id] = ast.literal_eval(node.value)
    return vals


def aryl_split(df: pd.DataFrame) -> Tuple[Dict[str, List[int]], Dict[str, object]]:
    eids, nids = ["e5"], ["nG"]
    ligs = sorted(df["ligand_name"].unique())[-11:]
    split = {"train": [], "substrate_oos": [], "catalyst_oos": [], "substrate_catalyst_oos": []}
    for i, row in df.iterrows():
        sub = row["electrophile_id"] in eids or row["nucleophile_id"] in nids
        cat = row["ligand_name"] in ligs
        split["substrate_catalyst_oos" if sub and cat else "substrate_oos" if sub else "catalyst_oos" if cat else "train"].append(i)
    return split, {"oos_electrophile_ids": eids, "oos_nucleophile_ids": nids, "oos_ligand_names": ligs}


def thiol_split(df: pd.DataFrame) -> Tuple[Dict[str, List[int]], Dict[str, object]]:
    return build_thiol_oos_split(df)


def aryl_bundle(repo: Path, repr_map: Dict[str, np.ndarray]) -> Dict[str, object]:
    data_dir = repo / "dataset" / "rxn_data"
    df = pd.read_csv(data_dir / "aryl-scope-ligand.csv")
    lig_cat_df = pd.read_csv(data_dir / "ligand_catalyst_map_of_aryl.csv")
    lig_cat = dict(zip(lig_cat_df["Ligand"], lig_cat_df["Catalyst"]))
    e = df["electrophile_smiles"].to_list()
    n = df["nucleophile_smiles"].to_list()
    ligand = df["ligand_smiles"].to_list()
    catalyst = [lig_cat[s] for s in ligand]
    split, split_def = aryl_split(df)
    return {
        "task": "Aryl scope yield prediction",
        "prefix": "aryl_scope",
        "dataset_file": str(data_dir / "aryl-scope-ligand.csv"),
        "y": df["yield"].to_numpy(),
        "test_size": 0.2,
        "split": split,
        "split_def": split_def,
        "n_samples": int(len(df)),
        "n_unique_smiles": int(len(set(e + n + ligand + catalyst))),
        "specs": [
            {
                "strategy_name": "aryl_scope_3dinfomax_electrophile_nucleophile_ligand",
                "encoded_molecules": ["electrophile_smiles", "nucleophile_smiles", "ligand_smiles"],
                "excluded_molecules": ["product_smiles", "pd_catalyst_smiles"],
                "x": feature_matrix(repr_map, [e, n, ligand]),
            },
            {
                "strategy_name": "aryl_scope_3dinfomax_electrophile_nucleophile_ligand_pd_catalyst",
                "encoded_molecules": ["electrophile_smiles", "nucleophile_smiles", "ligand_smiles", "pd_catalyst_smiles"],
                "excluded_molecules": ["product_smiles"],
                "x": feature_matrix(repr_map, [e, n, ligand, catalyst]),
            },
        ],
    }


def thiol_bundle(repo: Path, repr_map: Dict[str, np.ndarray]) -> Dict[str, object]:
    data_dir = repo / "dataset" / "rxn_data"
    df = pd.read_csv(data_dir / "NS_acetal_dataset_with_pdt.csv")
    imine = df["Imine"].to_list()
    thiol = df["Thiol"].to_list()
    cat = df["Catalyst"].to_list()
    split, split_def = thiol_split(df)
    return {
        "task": "Thiol addition / acetal dataset",
        "prefix": "thiol_addition",
        "dataset_file": str(data_dir / "NS_acetal_dataset_with_pdt.csv"),
        "y": df["ΔΔG"].to_numpy(),
        "test_size": 475 / 1075,
        "split": split,
        "split_def": split_def,
        "n_samples": int(len(df)),
        "n_unique_smiles": int(len(set(imine + thiol + cat))),
        "specs": [
            {
                "strategy_name": "thiol_addition_3dinfomax_imine_thiol_catalyst",
                "encoded_molecules": ["Imine", "Thiol", "Catalyst"],
                "excluded_molecules": ["Product"],
                "x": feature_matrix(repr_map, [imine, thiol, cat]),
            }
        ],
    }


def collect_smiles(repo: Path, dataset: str) -> List[str]:
    data_dir = repo / "dataset" / "rxn_data"
    smiles = []
    if dataset in {"aryl", "both"}:
        aryl_df = pd.read_csv(data_dir / "aryl-scope-ligand.csv")
        lig_cat_df = pd.read_csv(data_dir / "ligand_catalyst_map_of_aryl.csv")
        lig_cat = dict(zip(lig_cat_df["Ligand"], lig_cat_df["Catalyst"]))
        ligands = aryl_df["ligand_smiles"].to_list()
        smiles += aryl_df["electrophile_smiles"].to_list() + aryl_df["nucleophile_smiles"].to_list() + ligands
        smiles += [lig_cat[s] for s in ligands]
    if dataset in {"thiol", "both"}:
        thiol_df = pd.read_csv(data_dir / "NS_acetal_dataset_with_pdt.csv")
        smiles += thiol_df["Imine"].to_list() + thiol_df["Thiol"].to_list() + thiol_df["Catalyst"].to_list()
    return smiles


def evaluate_bundle(bundle: Dict[str, object], args: argparse.Namespace, metadata: Dict[str, object], out_dir: Path) -> None:
    random_results, oos_results, random_rows, oos_rows = [], [], [], []
    for spec in bundle["specs"]:
        x = spec["x"]
        for model_type in args.model_types:
            rec = {
                "strategy_name": spec["strategy_name"],
                "model_type": model_type,
                "repr_dim": int(x.shape[1] // len(spec["encoded_molecules"])),
                "reaction_feature_dim": int(x.shape[1]),
                "encoded_molecules": spec["encoded_molecules"],
                "excluded_molecules": spec["excluded_molecules"],
            }
            if args.mode in {"random", "both"}:
                print("Random: dataset=%s, strategy=%s, model_type=%s" % (bundle["prefix"], spec["strategy_name"], model_type))
                rows = run_random(x, bundle["y"], bundle["test_size"], args.seeds, model_type, args.n_estimators, args.random_seed)
                random_results.append(dict(rec, metrics_summary=summarize(rows), metrics_by_seed=rows))
                row = dict(rec, **summarize(rows))
                random_rows.append({k: v for k, v in row.items() if k not in {"encoded_molecules", "excluded_molecules"}})
            if args.mode in {"oos", "both"}:
                print("OOS: dataset=%s, strategy=%s, model_type=%s" % (bundle["prefix"], spec["strategy_name"], model_type))
                rows = run_oos(x, bundle["y"], bundle["split"], model_type, args.n_estimators, args.random_seed)
                oos_results.append(dict(rec, metrics_by_split=rows))
                for metric_row in rows:
                    row = dict(rec, **metric_row)
                    oos_rows.append({k: v for k, v in row.items() if k not in {"encoded_molecules", "excluded_molecules"}})

    if random_rows:
        pd.DataFrame(random_rows).sort_values(["strategy_name", "r2_mean", "mae_mean"], ascending=[True, False, True]).to_csv(
            out_dir / ("%s_3dinfomax_random_summary.csv" % bundle["prefix"]), index=False
        )
    if oos_rows:
        pd.DataFrame(oos_rows).sort_values(["split", "r2", "mae"], ascending=[True, False, True]).to_csv(
            out_dir / ("%s_3dinfomax_oos_summary.csv" % bundle["prefix"]), index=False
        )

    payload = dict(metadata)
    payload.update(
        {
            "task": bundle["task"],
            "dataset_file": bundle["dataset_file"],
            "n_samples": bundle["n_samples"],
            "n_unique_smiles": bundle["n_unique_smiles"],
            "seeds": args.seeds,
            "searched_model_types": args.model_types,
            "n_estimators": args.n_estimators,
            "random_seed": args.random_seed,
            "use_cache": args.use_cache,
            "random_results": random_results,
            "oos_split_definition": bundle["split_def"],
            "split_idx_sizes": {k: len(v) for k, v in bundle["split"].items()},
            "oos_results": oos_results,
        }
    )
    (out_dir / ("%s_3dinfomax_qspr.json" % bundle["prefix"])).write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    gen_desc_dir = repo / "notebook" / "gen_desc"
    out_dir = args.output_dir or (repo / "benchmark_results" / "3dinfomax_qspr")
    gen_desc_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_path or (gen_desc_dir / "3dinfomax_repr_map.npy")

    repr_map = build_3dinfomax_repr_map(
        smiles_list=collect_smiles(repo, args.dataset),
        infomax_root=args.infomax_root,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        cache_path=cache_path,
        batch_size=args.batch_size,
        device_name=args.device,
        use_cache=args.use_cache,
    )

    config_path = resolve_under_root(args.config, args.infomax_root)
    model_args = prepare_3dinfomax_args(args.infomax_root, args.config, args.checkpoint)
    metadata = {
        "model_name": "3DInfomax",
        "infomax_root": str(args.infomax_root.resolve()),
        "config": str(config_path.resolve()),
        "checkpoint": str(Path(model_args.checkpoint).resolve()),
        "descriptor_cache": str(cache_path),
        "batch_size": args.batch_size,
        "device": args.device,
    }

    if args.dataset in {"aryl", "both"}:
        evaluate_bundle(aryl_bundle(repo, repr_map), args, metadata, out_dir)
    if args.dataset in {"thiol", "both"}:
        evaluate_bundle(thiol_bundle(repo, repr_map), args, metadata, out_dir)
    print("Saved results to %s" % out_dir)


if __name__ == "__main__":
    main()
