#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
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

from oos_splits import thiol_split as build_thiol_oos_split
from tqdm.auto import tqdm

MODEL_SIZES = ["84m", "164m", "310m", "570m", "1.1B"]
MODEL_TYPES = ["ExtraTrees", "Ridge", "KernelRidge", "SVR"]
OOS_SPLITS = ["substrate_oos", "catalyst_oos", "substrate_catalyst_oos"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate UniMol v2 descriptors on random and OOS QSPR tasks.")
    p.add_argument("--dataset", choices=["aryl", "thiol", "both"], default="both")
    p.add_argument("--mode", choices=["random", "oos", "both"], default="both")
    p.add_argument("--model-sizes", nargs="+", default=MODEL_SIZES)
    p.add_argument("--model-types", nargs="+", default=MODEL_TYPES, choices=MODEL_TYPES)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--use-cache", dest="use_cache", action="store_true")
    p.add_argument("--no-cache", dest="use_cache", action="store_false")
    p.set_defaults(use_cache=True)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def sanitize_model_size(model_size):
    return model_size.lower().replace(".", "p")


def build_unimol_repr_map(model_size, smiles, cache_path, batch_size, use_cache):
    if use_cache and cache_path.exists():
        return np.load(cache_path, allow_pickle=True).item()
    from unimol_tools import UniMolRepr

    unimol_repr = UniMolRepr(data_type="molecule", remove_hs=False, model_name="unimolv2", model_size=model_size)
    unique_smiles = sorted(set(smiles))
    chunks = []
    for start in tqdm(range(0, len(unique_smiles), batch_size), desc="Uni-Mol batches"):
        batch = unique_smiles[start : start + batch_size]
        chunks.append(np.asarray(unimol_repr.get_repr(batch, return_atomic_reprs=True)["cls_repr"]))
    arr = np.concatenate(chunks, axis=0)
    repr_map = {s: arr[i] for i, s in enumerate(unique_smiles)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, repr_map)
    return repr_map


def feature_matrix(repr_map, groups):
    return np.concatenate([np.stack([repr_map[s] for s in group], axis=0) for group in groups], axis=1)


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
    train = np.asarray(split["train"], dtype=int)
    model = make_regressor(model_type, n_estimators, random_seed).fit(x[train], y[train])
    rows = []
    for split_name in OOS_SPLITS:
        idx = np.asarray(split[split_name], dtype=int)
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
    for i, row in df.iterrows():
        sub = row["electrophile_id"] in eids or row["nucleophile_id"] in nids
        cat = row["ligand_name"] in ligs
        split["substrate_catalyst_oos" if sub and cat else "substrate_oos" if sub else "catalyst_oos" if cat else "train"].append(i)
    return split, {"oos_electrophile_ids": eids, "oos_nucleophile_ids": nids, "oos_ligand_names": ligs}


def thiol_split(df):
    return build_thiol_oos_split(df)


def aryl_base(repo):
    df = pd.read_csv(repo / "dataset" / "rxn_data" / "aryl-scope-ligand.csv")
    e = df["electrophile_smiles"].to_list()
    n = df["nucleophile_smiles"].to_list()
    lig = df["ligand_smiles"].to_list()
    split, split_def = aryl_split(df)
    return {
        "task": "Aryl scope yield prediction",
        "prefix": "aryl_scope",
        "dataset_file": str(repo / "dataset" / "rxn_data" / "aryl-scope-ligand.csv"),
        "cache_prefix": "aryl_scope_unimolv2",
        "cache_suffix": "repr_map_no_product.npy",
        "sample_smiles": "Brc1ccc2c(ccn2Cc2ccccc2)c1",
        "y": df["yield"].to_numpy(),
        "test_size": 0.2,
        "split": split,
        "split_def": split_def,
        "smiles": lig + e + n,
        "groups": [e, n, lig],
        "encoded_molecules": ["electrophile_smiles", "nucleophile_smiles", "ligand_smiles"],
        "excluded_molecules": ["product_smiles"],
        "n_samples": int(len(df)),
    }


def thiol_base(repo):
    df = pd.read_csv(repo / "dataset" / "rxn_data" / "NS_acetal_dataset_with_pdt.csv")
    imine = df["Imine"].to_list()
    thiol = df["Thiol"].to_list()
    cat = df["Catalyst"].to_list()
    split, split_def = thiol_split(df)
    return {
        "task": "Thiol addition / acetal dataset",
        "prefix": "thiol_addition",
        "dataset_file": str(repo / "dataset" / "rxn_data" / "NS_acetal_dataset_with_pdt.csv"),
        "cache_prefix": "thiol_add_unimolv2",
        "cache_suffix": "repr_map_no_product.npy",
        "sample_smiles": cat[0],
        "y": df["ΔΔG"].to_numpy(),
        "test_size": 475 / 1075,
        "split": split,
        "split_def": split_def,
        "smiles": imine + thiol + cat,
        "groups": [imine, thiol, cat],
        "encoded_molecules": ["Imine", "Thiol", "Catalyst"],
        "excluded_molecules": ["Product"],
        "n_samples": int(len(df)),
    }


def eval_dataset(base, args, gen_desc_dir, out_dir):
    random_results, oos_results, random_rows, oos_rows = [], [], [], []
    for model_size in args.model_sizes:
        cache = gen_desc_dir / ("%s_%s_%s" % (base["cache_prefix"], sanitize_model_size(model_size), base["cache_suffix"]))
        repr_map = build_unimol_repr_map(model_size, base["smiles"], cache, args.batch_size, args.use_cache)
        x = feature_matrix(repr_map, base["groups"])
        common = {
            "model_name": "unimolv2", "model_size": model_size, "remove_hs": False,
            "repr_dim": int(x.shape[1] // len(base["groups"])), "reaction_feature_dim": int(x.shape[1]),
            "n_samples": base["n_samples"], "n_unique_smiles": int(len(repr_map)),
            "encoded_molecules": base["encoded_molecules"], "excluded_molecules": base["excluded_molecules"],
            "cache_path": str(cache), "sample_smiles": base["sample_smiles"],
            "sample_repr_shape": list(np.asarray(repr_map[base["sample_smiles"]]).shape),
        }
        for model_type in args.model_types:
            rec = dict(common, model_type=model_type)
            if args.mode in {"random", "both"}:
                print("Random: dataset=%s, model_size=%s, model_type=%s" % (base["prefix"], model_size, model_type))
                rows = run_random(x, base["y"], base["test_size"], args.seeds, model_type, args.n_estimators, args.random_seed)
                random_results.append(dict(rec, metrics_summary=summarize(rows), metrics_by_seed=rows))
                random_rows.append({"model_size": model_size, "model_type": model_type, "repr_dim": rec["repr_dim"], "reaction_feature_dim": rec["reaction_feature_dim"], **summarize(rows)})
            if args.mode in {"oos", "both"}:
                print("OOS: dataset=%s, model_size=%s, model_type=%s" % (base["prefix"], model_size, model_type))
                rows = run_oos(x, base["y"], base["split"], model_type, args.n_estimators, args.random_seed)
                oos_results.append(dict(rec, metrics_by_split=rows))
                oos_rows += [{"model_size": model_size, "model_type": model_type, "repr_dim": rec["repr_dim"], "reaction_feature_dim": rec["reaction_feature_dim"], **r} for r in rows]

    if random_rows:
        pd.DataFrame(random_rows).sort_values(["model_size", "r2_mean", "mae_mean"], ascending=[True, False, True]).to_csv(out_dir / ("%s_unimolv2_random_summary.csv" % base["prefix"]), index=False)
    if oos_rows:
        pd.DataFrame(oos_rows).sort_values(["split", "r2", "mae"], ascending=[True, False, True]).to_csv(out_dir / ("%s_unimolv2_oos_summary.csv" % base["prefix"]), index=False)

    payload = {
        "task": base["task"], "dataset_file": base["dataset_file"], "model_name": "unimolv2",
        "searched_model_sizes": args.model_sizes, "searched_model_types": args.model_types,
        "batch_size": args.batch_size, "seeds": args.seeds, "n_estimators": args.n_estimators,
        "random_seed": args.random_seed, "use_cache": args.use_cache,
        "random_results": random_results, "oos_split_definition": base["split_def"],
        "split_idx_sizes": {k: len(v) for k, v in base["split"].items()}, "oos_results": oos_results,
    }
    (out_dir / ("%s_unimolv2_qspr.json" % base["prefix"])).write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    gen_desc_dir = repo / "notebook" / "gen_desc"
    out_dir = args.output_dir or (repo / "benchmark_results" / "unimol_qspr")
    gen_desc_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset in {"aryl", "both"}:
        eval_dataset(aryl_base(repo), args, gen_desc_dir, out_dir)
    if args.dataset in {"thiol", "both"}:
        eval_dataset(thiol_base(repo), args, gen_desc_dir, out_dir)
    print("Saved results to %s" % out_dir)


if __name__ == "__main__":
    main()
