#!/usr/bin/env python3
import argparse
import ast
import json
from pathlib import Path

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

OOS_SPLITS = ["substrate_oos", "catalyst_oos", "substrate_catalyst_oos"]
MODEL_TYPES = ["ExtraTrees", "Ridge", "KernelRidge", "SVR"]


def metric(y, p):
    return {"r2": float(r2_score(y, p)), "mae": float(mean_absolute_error(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p)))}


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


def random_eval(x, y, test_size, seeds, model_type, n_estimators, random_seed):
    rows = []
    for seed in seeds:
        tx, vx, ty, vy = train_test_split(x, y, test_size=test_size, random_state=seed)
        pred = make_regressor(model_type, n_estimators, random_seed).fit(tx, ty).predict(vx)
        rows.append({"seed": int(seed), **metric(vy, pred)})
    return rows


def oos_eval(x, y, split, model_type, n_estimators, seed):
    train = np.asarray(split["train"])
    model = make_regressor(model_type, n_estimators, seed).fit(x[train], y[train])
    rows = []
    for s in OOS_SPLITS:
        idx = np.asarray(split[s])
        rows.append({"split": s, "n_samples": int(len(idx)), **metric(y[idx], model.predict(x[idx]))})
    return rows


def summary(rows):
    df = pd.DataFrame(rows)
    return {"r2_mean": float(df.r2.mean()), "r2_std": float(df.r2.std(ddof=1)), "mae_mean": float(df.mae.mean()), "mae_std": float(df.mae.std(ddof=1)), "rmse_mean": float(df.rmse.mean()), "rmse_std": float(df.rmse.std(ddof=1))}


def literal_constants(script_path, names):
    vals = {}
    for node in ast.parse(Path(script_path).read_text()).body:
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


def aryl_bundle(repo):
    df = pd.read_csv(repo / "dataset" / "rxn_data" / "aryl-scope-ligand.csv")
    fp_map = np.load(repo / "notebook" / "gen_desc" / "aryl_scope_rxn_smi_with_lig_rxnfp_map.npy", allow_pickle=True).item()
    rxn = [f"{e}.{n}.{l}>>{p}" for e, n, l, p in zip(df.electrophile_smiles, df.nucleophile_smiles, df.ligand_smiles, df.product_smiles)]
    split, split_def = aryl_split(df)
    return {
        "task": "Aryl scope yield prediction",
        "prefix": "aryl_scope",
        "strategy_name": "aryl_scope_rxnfp_all_with_ligand_product",
        "rxnfp_encoding": "electrophile.nucleophile.ligand>>product",
        "x": np.asarray([fp_map[s] for s in rxn]),
        "y": df["yield"].to_numpy(),
        "test_size": 0.2,
        "split": split,
        "split_def": split_def,
    }


def thiol_bundle(repo):
    df = pd.read_csv(repo / "dataset" / "rxn_data" / "NS_acetal_dataset_with_pdt.csv")
    fp_map = np.load(repo / "notebook" / "gen_desc" / "thiol_add_rxn_smi_with_cat_rxnfp_map.npy", allow_pickle=True).item()
    rxn = [f"{i}.{t}.{c}>>{p}" for i, t, c, p in zip(df.Imine, df.Thiol, df.Catalyst, df.Product)]
    split, split_def = thiol_split(df)
    return {
        "task": "Thiol addition / acetal dataset",
        "prefix": "thiol_addition",
        "strategy_name": "thiol_addition_rxnfp_all_with_catalyst_product",
        "rxnfp_encoding": "Imine.Thiol.Catalyst>>Product",
        "x": np.asarray([fp_map[s] for s in rxn]),
        "y": df["ΔΔG"].to_numpy(),
        "test_size": 475 / 1075,
        "split": split,
        "split_def": split_def,
    }


def evaluate(bundle, args, out):
    random_rows, oos_rows, results = [], [], []
    for model_type in args.model_types:
        print(f"Evaluating dataset={bundle['prefix']}, model_type={model_type}")
        rr = random_eval(bundle["x"], bundle["y"], bundle["test_size"], args.seeds, model_type, args.n_estimators, args.random_seed)
        oo = oos_eval(bundle["x"], bundle["y"], bundle["split"], model_type, args.n_estimators, args.random_seed)
        rec = {"strategy_name": bundle["strategy_name"], "model_type": model_type, "reaction_feature_dim": int(bundle["x"].shape[1]), "rxnfp_encoding": bundle["rxnfp_encoding"]}
        random_rows.append({**rec, **summary(rr)})
        oos_rows += [{**rec, **r} for r in oo]
        results.append({**rec, "random": {"metrics_summary": summary(rr), "metrics_by_seed": rr}, "oos": {"metrics_by_split": oo}})
    pd.DataFrame(random_rows).sort_values(["r2_mean", "mae_mean"], ascending=[False, True]).to_csv(out / f"{bundle['prefix']}_rxnfp_all_random_summary.csv", index=False)
    pd.DataFrame(oos_rows).sort_values(["split", "r2", "mae"], ascending=[True, False, True]).to_csv(out / f"{bundle['prefix']}_rxnfp_all_oos_summary.csv", index=False)
    payload = {"task": bundle["task"], "model_name": "rxnfp", "searched_model_types": args.model_types, "split_idx_sizes": {k: len(v) for k, v in bundle["split"].items()}, "oos_split_definition": bundle["split_def"], "results": results}
    (out / f"{bundle['prefix']}_rxnfp_all_qspr.json").write_text(json.dumps(payload, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["aryl", "thiol", "both"], default="both")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--model-types", nargs="+", default=MODEL_TYPES, choices=MODEL_TYPES)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = args.output_dir or (repo / "benchmark_results" / "rxnfp_all_qspr")
    out.mkdir(parents=True, exist_ok=True)
    if args.dataset in {"aryl", "both"}:
        evaluate(aryl_bundle(repo), args, out)
    if args.dataset in {"thiol", "both"}:
        evaluate(thiol_bundle(repo), args, out)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
