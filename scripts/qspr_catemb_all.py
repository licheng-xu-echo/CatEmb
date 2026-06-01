#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catemb import CatEmb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from oos_splits import thiol_split as build_thiol_oos_split

DIMS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
MODEL_ROOT = Path("/inspire/hdd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/8359-xulicheng/CatEmb/save_model")
MODEL_TYPES = ["ExtraTrees", "Ridge", "KernelRidge", "SVR"]
OOS_SPLITS = ["substrate_oos", "catalyst_oos", "substrate_catalyst_oos"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CatEmb descriptors for all reaction molecules on random and OOS QSPR tasks.")
    p.add_argument("--dataset", choices=["aryl", "thiol", "both"], default="both")
    p.add_argument("--mode", choices=["random", "oos", "both"], default="both")
    p.add_argument("--catemb-dims", nargs="+", type=int, default=DIMS)
    p.add_argument("--catemb-model-root", type=Path, default=MODEL_ROOT)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--model-types", nargs="+", default=MODEL_TYPES, choices=MODEL_TYPES)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def model_paths(root, dims, repo):
    paths = {}
    if root.exists():
        dirs = [p for p in root.iterdir() if p.is_dir()]
        for dim in dims:
            hit = [p for p in dirs if f"dim{dim}" in p.name and (p / "best_model.pt").exists() and (p / "full_params.npy").exists()]
            if hit:
                paths[dim] = sorted(hit)[0]
    local = repo / "catemb" / "model_path" / "dim32LN"
    if 32 in dims and 32 not in paths and local.exists():
        paths[32] = local
    missing = [d for d in dims if d not in paths]
    if missing:
        raise FileNotFoundError("Missing CatEmb models: %s" % missing)
    return paths


def desc_map(smiles, model_path, cache_path, batch_size):
    need = set(smiles)
    if cache_path.exists():
        m = np.load(cache_path, allow_pickle=True).item()
        if need <= set(m):
            return m
    calc = CatEmb(device="cpu", model_path=str(model_path))
    unique = sorted(need)
    arr = calc.gen_desc(unique, batch_size=batch_size)
    m = {s: np.asarray(arr[i]) for i, s in enumerate(unique)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, m)
    return m


def feature_matrix(desc, groups):
    return np.concatenate([np.stack([desc[s] for s in group], axis=0) for group in groups], axis=1)


def make_regressor(model_type, n_estimators, seed):
    if model_type == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    if model_type == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if model_type == "KernelRidge":
        return make_pipeline(StandardScaler(), KernelRidge(alpha=1.0, kernel="rbf"))
    if model_type == "SVR":
        return make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.1))
    raise ValueError("Unsupported model type: %s" % model_type)


def metrics(y, p):
    return {"r2": float(r2_score(y, p)), "mae": float(mean_absolute_error(y, p)), "rmse": float(root_mean_squared_error(y, p))}


def run_random(x, y, test_size, seeds, model_type, n_estimators, random_seed):
    rows = []
    for seed in seeds:
        tx, vx, ty, vy = train_test_split(x, y, test_size=test_size, random_state=seed)
        pred = make_regressor(model_type, n_estimators, random_seed).fit(tx, ty).predict(vx)
        rows.append({"seed": int(seed), **metrics(vy, pred)})
    return rows


def run_oos(x, y, split, model_type, n_estimators, random_seed):
    train = np.asarray(split["train"])
    model = make_regressor(model_type, n_estimators, random_seed).fit(x[train], y[train])
    rows = []
    for split_name in OOS_SPLITS:
        idx = np.asarray(split[split_name])
        rows.append({"split": split_name, "n_samples": int(len(idx)), **metrics(y[idx], model.predict(x[idx]))})
    return rows


def summarize(rows):
    df = pd.DataFrame(rows)
    return {
        "r2_mean": float(df.r2.mean()), "r2_std": float(df.r2.std(ddof=1)),
        "mae_mean": float(df.mae.mean()), "mae_std": float(df.mae.std(ddof=1)),
        "rmse_mean": float(df.rmse.mean()), "rmse_std": float(df.rmse.std(ddof=1)),
    }


def literal_constants(path, names):
    vals = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", "") in names:
            vals[node.targets[0].id] = ast.literal_eval(node.value)
    return vals


def aryl_split(df):
    eids, nids = ["e5"], ["nG"]
    ligs = sorted(df["ligand_name"].unique())[-11:]
    split = {"train": [], "substrate_oos": [], "catalyst_oos": [], "substrate_catalyst_oos": []}
    for i, r in df.iterrows():
        sub = r["electrophile_id"] in eids or r["nucleophile_id"] in nids
        cat = r["ligand_name"] in ligs
        split["substrate_catalyst_oos" if sub and cat else "substrate_oos" if sub else "catalyst_oos" if cat else "train"].append(i)
    return split, {"oos_electrophile_ids": eids, "oos_nucleophile_ids": nids, "oos_ligand_names": ligs}


def thiol_split(df):
    return build_thiol_oos_split(df)


def aryl_base(repo):
    df = pd.read_csv(repo / "dataset" / "rxn_data" / "aryl-scope-ligand.csv")
    lig_cat_df = pd.read_csv(repo / "dataset" / "rxn_data" / "ligand_catalyst_map_of_aryl.csv")
    lig_cat = dict(zip(lig_cat_df["Ligand"], lig_cat_df["Catalyst"]))
    e = df["electrophile_smiles"].to_list()
    n = df["nucleophile_smiles"].to_list()
    lig = df["ligand_smiles"].to_list()
    cat = [lig_cat[s] for s in lig]
    pdt = df["product_smiles"].to_list()
    split, split_def = aryl_split(df)
    return {
        "task": "Aryl scope yield prediction", "prefix": "aryl_scope", "cache_tag": "aryl",
        "strategy_name": "aryl_scope_catemb_all_with_product",
        "y": df["yield"].to_numpy(), "test_size": 0.2, "split": split, "split_def": split_def,
        "smiles": e + n + lig + cat + pdt, "groups": [e, n, lig, cat, pdt],
        "roles": ["electrophile_smiles", "nucleophile_smiles", "ligand_smiles", "pd_catalyst_smiles", "product_smiles"],
    }


def thiol_base(repo):
    df = pd.read_csv(repo / "dataset" / "rxn_data" / "NS_acetal_dataset_with_pdt.csv")
    imine = df["Imine"].to_list()
    thiol = df["Thiol"].to_list()
    cat = df["Catalyst"].to_list()
    pdt = df["Product"].to_list()
    split, split_def = thiol_split(df)
    return {
        "task": "Thiol addition / acetal dataset", "prefix": "thiol_addition", "cache_tag": "thiol",
        "strategy_name": "thiol_addition_catemb_all_with_product",
        "y": df["ΔΔG"].to_numpy(), "test_size": 475 / 1075, "split": split, "split_def": split_def,
        "smiles": imine + thiol + cat + pdt, "groups": [imine, thiol, cat, pdt],
        "roles": ["Imine", "Thiol", "Catalyst", "Product"],
    }


def evaluate_dataset(base, args, paths, gen_dir, out_dir):
    random_results, oos_results, random_rows, oos_rows = [], [], [], []
    for dim in args.catemb_dims:
        cache = gen_dir / ("catemb_all_%s_dim%s_repr_map.npy" % (base["cache_tag"], dim))
        desc = desc_map(base["smiles"], paths[dim], cache, args.batch_size)
        x = feature_matrix(desc, base["groups"])
        for model_type in args.model_types:
            rec = {
                "strategy_name": base["strategy_name"], "catemb_dim": dim, "model_type": model_type,
                "reaction_feature_dim": int(x.shape[1]), "catemb_roles": base["roles"],
            }
            if args.mode in {"random", "both"}:
                print("Random: dataset=%s, dim=%s, model_type=%s" % (base["prefix"], dim, model_type))
                rows = run_random(x, base["y"], base["test_size"], args.seeds, model_type, args.n_estimators, args.random_seed)
                random_results.append(dict(rec, metrics_summary=summarize(rows), metrics_by_seed=rows))
                random_rows.append(dict(rec, **summarize(rows)))
            if args.mode in {"oos", "both"}:
                print("OOS: dataset=%s, dim=%s, model_type=%s" % (base["prefix"], dim, model_type))
                rows = run_oos(x, base["y"], base["split"], model_type, args.n_estimators, args.random_seed)
                oos_results.append(dict(rec, metrics_by_split=rows))
                oos_rows += [dict(rec, **row) for row in rows]

    if random_rows:
        pd.DataFrame(random_rows).sort_values(["r2_mean", "mae_mean"], ascending=[False, True]).to_csv(out_dir / ("%s_catemb_all_random_summary.csv" % base["prefix"]), index=False)
    if oos_rows:
        pd.DataFrame(oos_rows).sort_values(["split", "r2", "mae"], ascending=[True, False, True]).to_csv(out_dir / ("%s_catemb_all_oos_summary.csv" % base["prefix"]), index=False)
    payload = {
        "task": base["task"], "model_name": "CatEmb", "catemb_dims": args.catemb_dims,
        "searched_model_types": args.model_types, "split_idx_sizes": {k: len(v) for k, v in base["split"].items()},
        "oos_split_definition": base["split_def"], "included_molecules": base["roles"],
        "random_results": random_results, "oos_results": oos_results,
    }
    (out_dir / ("%s_catemb_all_qspr.json" % base["prefix"])).write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    out_dir = args.output_dir or (repo / "benchmark_results" / "catemb_all_qspr")
    gen_dir = repo / "notebook" / "gen_desc"
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)
    paths = model_paths(args.catemb_model_root, args.catemb_dims, repo)
    if args.dataset in {"aryl", "both"}:
        evaluate_dataset(aryl_base(repo), args, paths, gen_dir, out_dir)
    if args.dataset in {"thiol", "both"}:
        evaluate_dataset(thiol_base(repo), args, paths, gen_dir, out_dir)
    print("Saved results to %s" % out_dir)


if __name__ == "__main__":
    main()
