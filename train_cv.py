"""5-fold stratified CV on the train split, then fit on all of train and score on test.

SMILES are featurized as 2048-bit Morgan fingerprints (radius 2, i.e. ECFP4) and
cached as .npz next to the CSVs so repeated runs / array jobs skip featurization.

Usage:
    python train_cv.py --model rf [--split Random] [--data_dir data/Hitgen_split]
                       [--output results] [--n_folds 5] [--seed 42]
"""
import argparse
import importlib
import json
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold

MODELS = ("mlp", "rf", "svm", "xgb")
FP_RADIUS = 2
FP_SIZE = 2048
TARGET_PRECISION = 0.95  # i.e. 5% false discovery rate, as in the paper


def featurize(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)
    X = np.zeros((len(smiles), FP_SIZE), dtype=np.uint8)
    valid = np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            X[i] = gen.GetFingerprintAsNumPy(mol)
            valid[i] = True
    return X, valid


def load_split(csv_path):
    """Return (X, y) for a CSV, featurizing on first use and caching to .npz."""
    cache = csv_path.with_name(f"{csv_path.stem}_morgan{FP_RADIUS}_{FP_SIZE}.npz")
    if cache.exists():
        data = np.load(cache)
        return data["X"], data["y"]

    df = pd.read_csv(csv_path)
    X, valid = featurize(df["SMILES"].tolist())
    if not valid.all():
        print(f"Dropping {(~valid).sum()} unparseable SMILES from {csv_path}")
    X, y = X[valid], df["label"].to_numpy()[valid]

    # Write to a per-process temp file then rename, so concurrent array jobs don't clash
    tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, X=X, y=y)
    os.replace(tmp, cache)
    return X, y


def predict_scores(model, X):
    """Return (scores, threshold) for computing ranking metrics and hard predictions."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1], 0.5
    if hasattr(model, "decision_function"):  # SVC without probability=True
        return model.decision_function(X), 0.0
    return model.predict(X, verbose=0).ravel(), 0.5  # keras


def fit_and_score(module, Xtr, ytr, Xev, yev):
    """Fit on train, return (model, scores on eval, timing info). No thresholding here."""
    model = module.build_model()

    start = time.perf_counter()
    model.fit(Xtr, ytr, **module.FIT_KWARGS)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    scores, _ = predict_scores(model, Xev)
    predict_seconds = time.perf_counter() - start

    info = {
        "n_train": len(ytr),
        "n_eval": len(yev),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }
    return model, scores, info


def classification_metrics(y, scores, threshold):
    preds = (scores >= threshold).astype(int)
    return {
        "balanced_accuracy": balanced_accuracy_score(y, preds),
        "mcc": matthews_corrcoef(y, preds),
        "f1": f1_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
    }


def precision_threshold(y, scores, target=TARGET_PRECISION):
    """Lowest score threshold whose precision on (y, scores) is >= target."""
    prec, _, thr = precision_recall_curve(y, scores)
    ok = prec[:-1] >= target  # drop the final dummy point (precision=1, no threshold)
    if not ok.any():
        raise SystemExit(f"No threshold reaches precision >= {target} on the CV scores "
                         f"(max precision {prec[:-1].max():.4f}); cannot set a hit threshold.")
    return float(thr[ok].min())


def format_metrics(m):
    return (f"BalAcc {m['balanced_accuracy']:.4f}, MCC {m['mcc']:.4f}, "
            f"F1 {m['f1']:.4f}, Recall {m['recall']:.4f}, Precision {m['precision']:.4f}")


def save_model(model, path_stem):
    if hasattr(model, "save") and not hasattr(model, "get_params"):  # keras
        model.save(f"{path_stem}.keras")
    else:
        joblib.dump(model, f"{path_stem}.joblib")


def main():
    parser = argparse.ArgumentParser(description="K-fold CV + held-out test for one model.")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--split", default="Random", help="Subdirectory of --data_dir (e.g. Random, Cluster)")
    parser.add_argument("--data_dir", default="data/Hitgen_split")
    parser.add_argument("--dataset", default="CK1d_HitGen_binary", help="CSV prefix before _train/_test")
    parser.add_argument("--output", "--output_dir", dest="output_dir", default="results")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total_start = time.perf_counter()
    module = importlib.import_module(f"models.{args.model}")

    split_dir = Path(args.data_dir) / args.split
    out_dir = Path(args.output_dir) / args.split / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    X, y = load_split(split_dir / f"{args.dataset}_train.csv")
    Xte, yte = load_split(split_dir / f"{args.dataset}_test.csv")
    load_seconds = time.perf_counter() - start
    print(f"Loaded train {X.shape}, test {Xte.shape} in {load_seconds:.1f}s")

    fold_rows, oof_parts = [], []
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        _, scores, info = fit_and_score(module, X[tr_idx], y[tr_idx], X[va_idx], y[va_idx])
        fold_rows.append({"fold": fold, **info})
        oof_parts.append(pd.DataFrame({"fold": fold, "train_index": va_idx,
                                       "label": y[va_idx], "score": scores}))
        print(f"[{args.model}] fold {fold}: fit {info['fit_seconds']:.1f}s")
        # Rewrite after every fold so partial results survive a job timeout
        pd.DataFrame(fold_rows).to_csv(out_dir / "cv_folds.csv", index=False)
        pd.concat(oof_parts).to_csv(out_dir / "cv_oof_scores.csv", index=False)

    # Hit threshold from the stacked held-out CV scores only (never from test)
    oof = pd.concat(oof_parts)
    threshold = precision_threshold(oof["label"].to_numpy(), oof["score"].to_numpy())
    print(f"[{args.model}] threshold for CV precision >= {TARGET_PRECISION}: {threshold:.6f}")

    # Per-fold metrics at that threshold. Optimistic: the threshold was tuned on these scores.
    for row, part in zip(fold_rows, oof_parts):
        row.update(classification_metrics(part["label"].to_numpy(), part["score"].to_numpy(), threshold))
    pd.DataFrame(fold_rows).to_csv(out_dir / "cv_folds.csv", index=False)
    folds = pd.DataFrame(fold_rows).drop(columns="fold")

    # Final model: fit on all of train, call test hits at the fixed CV-derived threshold
    final_model, test_scores, test_info = fit_and_score(module, X, y, Xte, yte)
    test_metrics = {**test_info, **classification_metrics(yte, test_scores, threshold)}
    save_model(final_model, out_dir / "final_model")
    print(f"[{args.model}] test: fit {test_metrics['fit_seconds']:.1f}s, "
          f"{format_metrics(test_metrics)}")

    summary = {
        "model": args.model,
        "split": args.split,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "target_precision": TARGET_PRECISION,
        "threshold": threshold,
        "cv_mean": folds.mean().to_dict(),
        "cv_std": folds.std().to_dict(),
        "cv_total_fit_seconds": folds["fit_seconds"].sum(),
        "test": test_metrics,
        "load_seconds": load_seconds,
        "total_seconds": time.perf_counter() - total_start,
        "host": platform.node(),
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
