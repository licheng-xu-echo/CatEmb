#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from catemb import CatEmb
from oos_splits import OOS_SPLITS, thiol_split as build_thiol_oos_split


DEFAULT_CATEMB_DIMS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_CATEMB_MODEL_ROOT = None
MODEL_TYPES = ["ExtraTrees", "Ridge", "KernelRidge", "SVR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate rxnfp + CatEmb QSPR on the aryl scope and thiol addition datasets."
    )
    parser.add_argument(
        "--catemb-dims",
        nargs="+",
        type=int,
        default=DEFAULT_CATEMB_DIMS,
        help="CatEmb embedding dimensions to evaluate.",
    )
    parser.add_argument(
        "--catemb-model-root",
        type=Path,
        default=DEFAULT_CATEMB_MODEL_ROOT,
        help="Directory containing CatEmb model checkpoints. Defaults to catemb/model_path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for CatEmb descriptor generation.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(10)),
        help="Random seeds for repeated train/test split evaluation.",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=MODEL_TYPES,
        choices=MODEL_TYPES,
        help="Regressor types to evaluate.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Number of trees in ExtraTreesRegressor.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save per-dataset outputs. Defaults to benchmark_results/rxnfp_catemb_qspr",
    )
    parser.add_argument(
        "--output-oos-csv",
        type=Path,
        default=None,
        help="Optional legacy path to save a combined OOS summary CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional legacy path to save the combined benchmark JSON.",
    )
    return parser.parse_args()


def resolve_catemb_model_paths(
    repo_root: Path,
    model_root: Path | None,
    dims: list[int],
) -> dict[int, Path]:
    root = model_root or (repo_root / "catemb" / "model_path")
    resolved_paths: dict[int, Path] = {}

    for dim in dims:
        candidates = [root / f"dim{dim}LN", root / f"dim{dim}"]
        if root.exists():
            candidates.extend(
                path for path in sorted(root.iterdir()) if path.is_dir() and f"dim{dim}" in path.name
            )
        for candidate in candidates:
            if (candidate / "best_model.pt").exists() and (candidate / "full_params.npy").exists():
                resolved_paths[dim] = candidate
                break

    missing_dims = [dim for dim in dims if dim not in resolved_paths]
    if missing_dims:
        raise FileNotFoundError(
            f"Could not resolve CatEmb model paths in {root} for dimensions: {missing_dims}"
        )

    return resolved_paths


def build_catemb_repr_map(
    catemb_calc: CatEmb,
    smiles_list: list[str],
    batch_size: int,
) -> dict[str, np.ndarray]:
    unique_smiles = sorted(set(smiles_list))
    desc_arr = catemb_calc.gen_desc(unique_smiles, batch_size=batch_size)
    return {smiles: np.asarray(desc_arr[i]) for i, smiles in enumerate(unique_smiles)}


def stack_repr_from_map(
    repr_map: dict[str, np.ndarray],
    smiles_list: list[str],
) -> np.ndarray:
    return np.stack([repr_map[smiles] for smiles in smiles_list], axis=0)


def make_regressor(model_type: str, n_estimators: int, random_seed: int):
    if model_type == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=n_estimators, random_state=random_seed, n_jobs=-1)
    if model_type == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if model_type == "KernelRidge":
        return make_pipeline(StandardScaler(), KernelRidge(alpha=1.0, kernel="rbf"))
    if model_type == "SVR":
        return make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.1))
    raise ValueError(f"Unsupported model type: {model_type}")


def run_qspr(
    feature_arr: np.ndarray,
    label_arr: np.ndarray,
    test_size: float,
    seeds: list[int],
    model_type: str,
    n_estimators: int,
    random_seed: int,
) -> list[dict[str, float]]:
    metrics = []
    for seed in seeds:
        train_x, test_x, train_y, test_y = train_test_split(
            feature_arr,
            label_arr,
            test_size=test_size,
            random_state=seed,
        )
        model = make_regressor(model_type, n_estimators=n_estimators, random_seed=random_seed)
        model.fit(train_x, train_y)
        test_p = model.predict(test_x)
        metrics.append(
            {
                "seed": int(seed),
                "r2": float(r2_score(test_y, test_p)),
                "mae": float(mean_absolute_error(test_y, test_p)),
                "rmse": float(root_mean_squared_error(test_y, test_p)),
            }
        )
    return metrics


def run_oos_qspr(
    feature_arr: np.ndarray,
    label_arr: np.ndarray,
    split_idx_map: dict[str, list[int]],
    model_type: str,
    n_estimators: int,
    random_seed: int,
) -> list[dict[str, object]]:
    train_idx = np.asarray(split_idx_map["train"], dtype=int)
    model = make_regressor(model_type, n_estimators=n_estimators, random_seed=random_seed)
    model.fit(feature_arr[train_idx], label_arr[train_idx])
    rows = []
    for split_name in OOS_SPLITS:
        idx = np.asarray(split_idx_map[split_name], dtype=int)
        pred = model.predict(feature_arr[idx])
        rows.append(
            {
                "split": split_name,
                "n_samples": int(len(idx)),
                "r2": float(r2_score(label_arr[idx], pred)),
                "mae": float(mean_absolute_error(label_arr[idx], pred)),
                "rmse": float(root_mean_squared_error(label_arr[idx], pred)),
            }
        )
    return rows


def build_aryl_split(aryl_df: pd.DataFrame) -> tuple[dict[str, list[int]], dict[str, object]]:
    oos_electrophile_ids = ["e5"]
    oos_nucleophile_ids = ["nG"]
    oos_ligand_names = sorted(aryl_df["ligand_name"].unique())[-11:]
    split_idx_map = {"train": [], "substrate_oos": [], "catalyst_oos": [], "substrate_catalyst_oos": []}
    for i, row in aryl_df.iterrows():
        is_oos_substrate = row["electrophile_id"] in oos_electrophile_ids or row["nucleophile_id"] in oos_nucleophile_ids
        is_oos_catalyst = row["ligand_name"] in oos_ligand_names
        split_idx_map["substrate_catalyst_oos" if is_oos_substrate and is_oos_catalyst else "substrate_oos" if is_oos_substrate else "catalyst_oos" if is_oos_catalyst else "train"].append(i)
    split_def = {"oos_electrophile_ids": oos_electrophile_ids, "oos_nucleophile_ids": oos_nucleophile_ids, "oos_ligand_names": oos_ligand_names}
    return split_idx_map, split_def


def summarize_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    metrics_df = pd.DataFrame(metrics)
    return {
        "r2_mean": float(metrics_df["r2"].mean()),
        "r2_std": float(metrics_df["r2"].std(ddof=1)),
        "mae_mean": float(metrics_df["mae"].mean()),
        "mae_std": float(metrics_df["mae"].std(ddof=1)),
        "rmse_mean": float(metrics_df["rmse"].mean()),
        "rmse_std": float(metrics_df["rmse"].std(ddof=1)),
    }


def evaluate_aryl_scope(
    repo_root: Path,
    catemb_calc: CatEmb,
    catemb_model_dim: int,
    catemb_model_path: Path,
    batch_size: int,
    seeds: list[int],
    model_types: list[str],
    n_estimators: int,
    random_seed: int,
) -> list[dict[str, object]]:
    data_dir = repo_root / "dataset" / "rxn_data"
    gen_desc_dir = repo_root / "notebook" / "gen_desc"

    aryl_df = pd.read_csv(data_dir / "aryl-scope-ligand.csv")
    lig_cat_df = pd.read_csv(data_dir / "ligand_catalyst_map_of_aryl.csv")
    lig_cat_map = {
        ligand: catalyst
        for ligand, catalyst in zip(lig_cat_df["Ligand"].to_list(), lig_cat_df["Catalyst"].to_list())
    }

    aryl_ligand = aryl_df["ligand_smiles"].to_list()
    aryl_rct1 = aryl_df["electrophile_smiles"].to_list()
    aryl_rct2 = aryl_df["nucleophile_smiles"].to_list()
    aryl_pdt = aryl_df["product_smiles"].to_list()
    aryl_label = aryl_df["yield"].to_numpy()
    aryl_cat = [lig_cat_map[ligand] for ligand in aryl_ligand]
    split_idx_map, split_def = build_aryl_split(aryl_df)

    rxnfp_wo_lig_map = np.load(
        gen_desc_dir / "aryl_scope_rxn_smi_wo_lig_rxnfp_map.npy",
        allow_pickle=True,
    ).item()
    rxn_smi_wo_lig = [
        f"{rct1}.{rct2}>>{pdt}"
        for rct1, rct2, pdt in zip(aryl_rct1, aryl_rct2, aryl_pdt)
    ]
    rxn_fp_arr = np.array([rxnfp_wo_lig_map[rxn_smi] for rxn_smi in rxn_smi_wo_lig])

    ligand_repr_map = build_catemb_repr_map(
        catemb_calc=catemb_calc,
        smiles_list=aryl_ligand,
        batch_size=batch_size,
    )
    catalyst_repr_map = build_catemb_repr_map(
        catemb_calc=catemb_calc,
        smiles_list=aryl_cat,
        batch_size=batch_size,
    )
    lig_desc = stack_repr_from_map(ligand_repr_map, aryl_ligand)
    cat_desc = stack_repr_from_map(catalyst_repr_map, aryl_cat)

    strategy_specs = [
        {
            "strategy_name": "aryl_scope_rxnfp_plus_catemb_ligand",
            "feature_arr": np.concatenate([rxn_fp_arr, lig_desc], axis=1),
            "catemb_roles": ["ligand_smiles"],
        },
        {
            "strategy_name": "aryl_scope_rxnfp_plus_catemb_ligand_pd_catalyst",
            "feature_arr": np.concatenate([rxn_fp_arr, lig_desc, cat_desc], axis=1),
            "catemb_roles": ["ligand_smiles", "pd_catalyst_smiles"],
        },
    ]

    results = []
    for spec in strategy_specs:
        for model_type in model_types:
            print(f"Evaluating dim={catemb_model_dim}, strategy={spec['strategy_name']}, model_type={model_type}")
            metrics = run_qspr(
                feature_arr=spec["feature_arr"],
                label_arr=aryl_label,
                test_size=0.2,
                seeds=seeds,
                model_type=model_type,
                n_estimators=n_estimators,
                random_seed=random_seed,
            )
            oos_metrics = run_oos_qspr(
                feature_arr=spec["feature_arr"],
                label_arr=aryl_label,
                split_idx_map=split_idx_map,
                model_type=model_type,
                n_estimators=n_estimators,
                random_seed=random_seed,
            )
            results.append(
                {
                    "task": "Aryl scope yield prediction",
                    "strategy_name": spec["strategy_name"],
                    "model_type": model_type,
                    "dataset_file": str(data_dir / "aryl-scope-ligand.csv"),
                    "rxnfp_feature_file": str(gen_desc_dir / "aryl_scope_rxn_smi_wo_lig_rxnfp_map.npy"),
                    "catemb_model_dim": int(catemb_model_dim),
                    "catemb_model_path": str(catemb_model_path.resolve()),
                    "catemb_cache_used": False,
                    "n_samples": int(len(aryl_df)),
                    "rxnfp_dim": int(rxn_fp_arr.shape[1]),
                    "catemb_dim": int(lig_desc.shape[1]),
                    "reaction_feature_dim": int(spec["feature_arr"].shape[1]),
                    "rxnfp_representation": "reactant-only rxnfp without ligand",
                    "catemb_roles": spec["catemb_roles"],
                    "oos_split_definition": split_def,
                    "split_idx_sizes": {split: len(idx) for split, idx in split_idx_map.items()},
                    "metrics_summary": summarize_metrics(metrics),
                    "metrics_by_seed": metrics,
                    "metrics_by_split": oos_metrics,
                }
            )
    return results


def evaluate_thiol_addition(
    repo_root: Path,
    catemb_calc: CatEmb,
    catemb_model_dim: int,
    catemb_model_path: Path,
    batch_size: int,
    seeds: list[int],
    model_types: list[str],
    n_estimators: int,
    random_seed: int,
) -> list[dict[str, object]]:
    data_dir = repo_root / "dataset" / "rxn_data"
    gen_desc_dir = repo_root / "notebook" / "gen_desc"

    thiol_df = pd.read_csv(data_dir / "NS_acetal_dataset_with_pdt.csv")
    imine_lst = thiol_df["Imine"].to_list()
    thiol_lst = thiol_df["Thiol"].to_list()
    cat_lst = thiol_df["Catalyst"].to_list()
    pdt_lst = thiol_df["Product"].to_list()
    thiol_label = thiol_df["ΔΔG"].to_numpy()
    split_idx_map, split_def = build_thiol_oos_split(thiol_df)

    rxnfp_wo_cat_map = np.load(
        gen_desc_dir / "thiol_add_rxn_smi_wo_cat_rxnfp_map.npy",
        allow_pickle=True,
    ).item()
    rxn_smi_wo_cat = [
        f"{imine}.{thiol}>>{pdt}"
        for imine, thiol, pdt in zip(imine_lst, thiol_lst, pdt_lst)
    ]
    rxn_fp_arr = np.array([rxnfp_wo_cat_map[rxn_smi] for rxn_smi in rxn_smi_wo_cat])

    catalyst_repr_map = build_catemb_repr_map(
        catemb_calc=catemb_calc,
        smiles_list=cat_lst,
        batch_size=batch_size,
    )
    cat_desc = stack_repr_from_map(catalyst_repr_map, cat_lst)
    feature_arr = np.concatenate([rxn_fp_arr, cat_desc], axis=1)

    results = []
    for model_type in model_types:
        print(f"Evaluating dim={catemb_model_dim}, strategy=thiol_addition_rxnfp_plus_catemb_catalyst, model_type={model_type}")
        metrics = run_qspr(
            feature_arr=feature_arr,
            label_arr=thiol_label,
            test_size=475 / 1075,
            seeds=seeds,
            model_type=model_type,
            n_estimators=n_estimators,
            random_seed=random_seed,
        )
        oos_metrics = run_oos_qspr(
            feature_arr=feature_arr,
            label_arr=thiol_label,
            split_idx_map=split_idx_map,
            model_type=model_type,
            n_estimators=n_estimators,
            random_seed=random_seed,
        )
        results.append({
            "task": "Thiol addition / acetal dataset",
            "strategy_name": "thiol_addition_rxnfp_plus_catemb_catalyst",
            "model_type": model_type,
            "dataset_file": str(data_dir / "NS_acetal_dataset_with_pdt.csv"),
            "rxnfp_feature_file": str(gen_desc_dir / "thiol_add_rxn_smi_wo_cat_rxnfp_map.npy"),
            "catemb_model_dim": int(catemb_model_dim),
            "catemb_model_path": str(catemb_model_path.resolve()),
            "catemb_cache_used": False,
            "n_samples": int(len(thiol_df)),
            "rxnfp_dim": int(rxn_fp_arr.shape[1]),
            "catemb_dim": int(cat_desc.shape[1]),
            "reaction_feature_dim": int(feature_arr.shape[1]),
            "rxnfp_representation": "reactant-only rxnfp without catalyst",
            "catemb_roles": ["Catalyst"],
            "oos_split_definition": split_def,
            "split_idx_sizes": {split: len(idx) for split, idx in split_idx_map.items()},
            "metrics_summary": summarize_metrics(metrics),
            "metrics_by_seed": metrics,
            "metrics_by_split": oos_metrics,
        })
    return results


def compact_result(result: dict[str, object]) -> dict[str, object]:
    return {
        "strategy_name": result["strategy_name"],
        "catemb_dim": result["catemb_model_dim"],
        "model_type": result["model_type"],
        "reaction_feature_dim": result["reaction_feature_dim"],
        "rxnfp_representation": result["rxnfp_representation"],
        "catemb_roles": result["catemb_roles"],
    }


def write_dataset_outputs(
    results: list[dict[str, object]],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    task_map = {
        "Aryl scope yield prediction": "aryl_scope",
        "Thiol addition / acetal dataset": "thiol_addition",
    }

    for task, prefix in task_map.items():
        task_results = [result for result in results if result["task"] == task]
        if not task_results:
            continue

        random_results = []
        oos_results = []
        random_rows = []
        oos_rows = []
        for result in task_results:
            rec = compact_result(result)
            random_results.append(
                {
                    **rec,
                    "metrics_summary": result["metrics_summary"],
                    "metrics_by_seed": result["metrics_by_seed"],
                }
            )
            oos_results.append({**rec, "metrics_by_split": result["metrics_by_split"]})
            random_rows.append({**rec, **result["metrics_summary"]})
            oos_rows.extend({**rec, **row} for row in result["metrics_by_split"])

        pd.DataFrame(random_rows).sort_values(
            ["strategy_name", "r2_mean", "mae_mean"],
            ascending=[True, False, True],
        ).to_csv(out_dir / f"{prefix}_rxnfp_catemb_random_summary.csv", index=False)

        pd.DataFrame(oos_rows).sort_values(
            ["split", "r2", "mae"],
            ascending=[True, False, True],
        ).to_csv(out_dir / f"{prefix}_rxnfp_catemb_oos_summary.csv", index=False)

        first = task_results[0]
        payload = {
            "task": task,
            "model_name": "rxnfp+CatEmb",
            "catemb_dims": args.catemb_dims,
            "searched_model_types": args.model_types,
            "split_idx_sizes": first["split_idx_sizes"],
            "oos_split_definition": first["oos_split_definition"],
            "random_results": random_results,
            "oos_results": oos_results,
        }
        (out_dir / f"{prefix}_rxnfp_catemb_qspr.json").write_text(json.dumps(payload, indent=2))


def write_legacy_outputs(
    results: list[dict[str, object]],
    output_json: Path | None,
    output_oos_csv: Path | None,
    args: argparse.Namespace,
    catemb_model_paths: dict[int, Path],
) -> None:
    if output_oos_csv:
        oos_rows = []
        for result in results:
            for metric in result["metrics_by_split"]:
                oos_rows.append(
                    {
                        "task": result["task"],
                        "strategy_name": result["strategy_name"],
                        "model_type": result["model_type"],
                        "catemb_dim": result["catemb_model_dim"],
                        "reaction_feature_dim": result["reaction_feature_dim"],
                        **metric,
                    }
                )
        pd.DataFrame(oos_rows).sort_values(
            ["task", "split", "r2", "mae"],
            ascending=[True, True, False, True],
        ).to_csv(output_oos_csv, index=False)

    if output_json:
        payload = {
            "catemb_dims": args.catemb_dims,
            "catemb_model_root": str(args.catemb_model_root),
            "catemb_model_paths": {str(dim): str(path.resolve()) for dim, path in catemb_model_paths.items()},
            "batch_size": args.batch_size,
            "seeds": args.seeds,
            "searched_model_types": args.model_types,
            "n_estimators": args.n_estimators,
            "random_seed": args.random_seed,
            "catemb_cache_used": False,
            "oos_summary_csv": str(output_oos_csv) if output_oos_csv else None,
            "results": results,
        }
        output_json.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = args.output_dir or (repo_root / "benchmark_results" / "rxnfp_catemb_qspr")
    out_dir.mkdir(parents=True, exist_ok=True)

    catemb_model_paths = resolve_catemb_model_paths(
        repo_root=repo_root,
        model_root=args.catemb_model_root,
        dims=args.catemb_dims,
    )

    results = []
    for dim in args.catemb_dims:
        model_path = catemb_model_paths[dim]
        print(f"Evaluating CatEmb dim={dim}: {model_path}")
        catemb_calc = CatEmb(device="cpu", model_path=str(model_path))

        aryl_results = evaluate_aryl_scope(
            repo_root=repo_root,
            catemb_calc=catemb_calc,
            catemb_model_dim=dim,
            catemb_model_path=model_path,
            batch_size=args.batch_size,
            seeds=args.seeds,
            model_types=args.model_types,
            n_estimators=args.n_estimators,
            random_seed=args.random_seed,
        )
        thiol_results = evaluate_thiol_addition(
            repo_root=repo_root,
            catemb_calc=catemb_calc,
            catemb_model_dim=dim,
            catemb_model_path=model_path,
            batch_size=args.batch_size,
            seeds=args.seeds,
            model_types=args.model_types,
            n_estimators=args.n_estimators,
            random_seed=args.random_seed,
        )
        results.extend(aryl_results)
        results.extend(thiol_results)

    write_dataset_outputs(results, out_dir, args)
    write_legacy_outputs(results, args.output_json, args.output_oos_csv, args, catemb_model_paths)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
