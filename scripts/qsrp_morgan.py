#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from oos_splits import OOS_CAT, OOS_IMINE, OOS_THIOL, thiol_split as build_thiol_oos_split

OOS_SPLITS = ["substrate_oos", "catalyst_oos", "substrate_catalyst_oos"]
MODEL_TYPES = ["ExtraTrees", "Ridge", "KernelRidge", "SVR"]


def morgan_map(smiles, radius, n_bits):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = {}
    for smi in sorted(set(smiles)):
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(Chem.MolFromSmiles(smi)), arr)
        out[smi] = arr
    return out


def features(fp_map, groups):
    return np.concatenate([np.stack([fp_map[s] for s in g]) for g in groups], axis=1)


def metrics(y, p):
    return {
        "r2": float(r2_score(y, p)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(root_mean_squared_error(y, p)),
    }


def make_regressor(model_type, n_estimators, seed):
    if model_type == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    if model_type == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if model_type == "KernelRidge":
        return make_pipeline(StandardScaler(), KernelRidge(alpha=1.0, kernel="rbf"))
    if model_type == "SVR":
        return make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.1))
    raise ValueError(f"Unsupported model type: {model_type}")


def run_random(x, y, test_size, seeds, model_type, n_estimators, random_seed):
    rows = []
    for seed in seeds:
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, random_state=seed)
        model = make_regressor(model_type, n_estimators, random_seed).fit(train_x, train_y)
        rows.append({"seed": int(seed), **metrics(test_y, model.predict(test_x))})
    return rows


def summarize(rows):
    df = pd.DataFrame(rows)
    return {k: float(v) for k, v in {
        "r2_mean": df.r2.mean(), "r2_std": df.r2.std(ddof=1),
        "mae_mean": df.mae.mean(), "mae_std": df.mae.std(ddof=1),
        "rmse_mean": df.rmse.mean(), "rmse_std": df.rmse.std(ddof=1),
    }.items()}


def run_oos(x, y, split_idx, model_type, n_estimators, seed):
    train = np.asarray(split_idx["train"])
    model = make_regressor(model_type, n_estimators, seed).fit(x[train], y[train])
    rows = []
    for split in OOS_SPLITS:
        idx = np.asarray(split_idx[split])
        rows.append({"split": split, "n_samples": int(len(idx)), **metrics(y[idx], model.predict(x[idx]))})
    return rows


def literal_constants(script_path, names):
    mod = ast.parse(Path(script_path).read_text())
    vals = {}
    for node in mod.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", "") in names:
            vals[node.targets[0].id] = ast.literal_eval(node.value)
    return vals


def thiol_bundle(root):
    df = pd.read_csv(root / "dataset" / "rxn_data" / "NS_acetal_dataset_with_pdt.csv")
    imine, thiol, cat = df["Imine"].to_list(), df["Thiol"].to_list(), df["Catalyst"].to_list()
    split, split_def = build_thiol_oos_split(df)
    return {
        "task": "Thiol addition / acetal dataset",
        "file_prefix": "thiol_addition",
        "strategy_name": "thiol_addition_morgan_imine_thiol_catalyst",
        "y": df["ΔΔG"].to_numpy(),
        "fp_smiles": imine + thiol + cat,
        "specs": [("thiol_addition_morgan_imine_thiol_catalyst", [imine, thiol, cat], None)],
        "test_size": 475 / 1075,
        "split": split,
        "split_def": split_def,
    }


def aryl_bundle(root):
    data_dir = root / "dataset" / "rxn_data"
    df = pd.read_csv(data_dir / "aryl-scope-ligand.csv")
    lig_cat_df = pd.read_csv(data_dir / "ligand_catalyst_map_of_aryl.csv")
    lig_cat = dict(zip(lig_cat_df["Ligand"], lig_cat_df["Catalyst"]))
    e, n, l = df["electrophile_smiles"].to_list(), df["nucleophile_smiles"].to_list(), df["ligand_smiles"].to_list()
    c = [lig_cat[s] for s in l]
    oos_electrophile_ids = ["e5"]
    oos_nucleophile_ids = ["nG"]
    oos_ligand_names = sorted(df["ligand_name"].unique())[-11:]
    split = {"train": [], **{k: [] for k in OOS_SPLITS}}
    for i, row in df.iterrows():
        sub = row["electrophile_id"] in oos_electrophile_ids or row["nucleophile_id"] in oos_nucleophile_ids
        is_cat = row["ligand_name"] in oos_ligand_names
        split["substrate_catalyst_oos" if sub and is_cat else "substrate_oos" if sub else "catalyst_oos" if is_cat else "train"].append(i)
    return {
        "task": "Aryl scope yield prediction",
        "file_prefix": "aryl_scope",
        "y": df["yield"].to_numpy(),
        "fp_smiles": e + n + l + c,
        "specs": [
            ("aryl_scope_morgan_electrophile_nucleophile_ligand", [e, n, l], ["electrophile_smiles", "nucleophile_smiles", "ligand_smiles"]),
            ("aryl_scope_morgan_electrophile_nucleophile_ligand_pd_catalyst", [e, n, l, c], ["electrophile_smiles", "nucleophile_smiles", "ligand_smiles", "pd_catalyst_smiles"]),
        ],
        "test_size": 0.2,
        "split": split,
        "split_def": {"oos_electrophile_ids": oos_electrophile_ids, "oos_nucleophile_ids": oos_nucleophile_ids, "oos_ligand_names": oos_ligand_names},
    }


def evaluate_bundle(bundle, args, out_dir):
    fp = morgan_map(bundle["fp_smiles"], args.radius, args.n_bits)
    random_results, oos_results, random_rows, oos_rows = [], [], [], []
    for strategy, groups, roles in bundle["specs"]:
        x = features(fp, groups)
        for model_type in args.model_types:
            print(f"Evaluating strategy={strategy}, model_type={model_type}")
            seeds = run_random(x, bundle["y"], bundle["test_size"], args.seeds, model_type, args.n_estimators, args.random_seed)
            oos = run_oos(x, bundle["y"], bundle["split"], model_type, args.n_estimators, args.random_seed)
            rec = {"strategy_name": strategy, "model_type": model_type}
            if roles is not None:
                rec["encoded_molecules"] = roles
            rec["reaction_feature_dim"] = int(x.shape[1])
            random_results.append({**rec, "metrics_summary": summarize(seeds), "metrics_by_seed": seeds})
            oos_results.append({**rec, "metrics_by_split": oos})
            random_rec = {k: v for k, v in rec.items() if k != "encoded_molecules"}
            random_rows.append({**random_rec, **summarize(seeds)})
            oos_rows += [{**rec, **r} for r in oos]
    payload = {
        "task": bundle["task"],
        "model_name": "MorganFingerprint",
        "radius": args.radius,
        "n_bits": args.n_bits,
        "seeds": args.seeds,
        "searched_model_types": args.model_types,
        "n_estimators": args.n_estimators,
        "random_results": random_results,
        "oos_split_definition": bundle["split_def"],
        "split_idx_sizes": {k: len(v) for k, v in bundle["split"].items()},
        "oos_results": oos_results,
    }
    if "strategy_name" in bundle:
        payload["strategy_name"] = bundle["strategy_name"]
        payload["reaction_feature_dim"] = random_results[0]["reaction_feature_dim"]
    (out_dir / f"{bundle['file_prefix']}_morgan_qspr.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(oos_rows).sort_values(["split", "r2", "mae"], ascending=[True, False, True]).to_csv(out_dir / f"{bundle['file_prefix']}_morgan_oos_summary.csv", index=False)
    random_sort = ["strategy_name", "r2_mean", "mae_mean"] if bundle["file_prefix"] == "aryl_scope" else ["r2_mean", "mae_mean"]
    pd.DataFrame(random_rows).sort_values(random_sort, ascending=[True, False, True] if len(random_sort) == 3 else [False, True]).to_csv(out_dir / f"{bundle['file_prefix']}_morgan_random_summary.csv", index=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["aryl", "thiol", "both"], default="both")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--model-types", nargs="+", default=MODEL_TYPES, choices=MODEL_TYPES)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--random-seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "benchmark_results" / "morgan_qspr"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundles = []
    if args.dataset in {"aryl", "both"}:
        bundles.append(aryl_bundle(root))
    if args.dataset in {"thiol", "both"}:
        bundles.append(thiol_bundle(root))
    for bundle in bundles:
        evaluate_bundle(bundle, args, out_dir)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
