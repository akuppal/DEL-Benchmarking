"""5-fold stratified CV on the train split, then fit on all of train and score on test,
for Chemprop v2 (a directed message-passing neural network on the molecular graph).

Matches train_cv.py so the results line up with the other models:
  * the same rows (SMILES that RDKit cannot parse are dropped, as in train_cv.py),
  * the same StratifiedKFold folds (same n_folds and seed),
  * the same hit threshold: the lowest score reaching 95% precision on the stacked
    out-of-fold CV scores, applied unchanged to the test set,
  * the same metrics and output files in results/<split>/chemprop/.

Chemprop needs a validation set for early stopping. It is a stratified --val_frac
slice of each training fold (and of the full train set for the final model), never
the evaluation fold or the test set.

Finished folds are saved and skipped on rerun, so a job that hits its time limit can
just be resubmitted with the same command.

Usage:
    python train_chemprop.py [--split Random] [--epochs 50] [--batch_size 64]
    python train_chemprop.py --max_rows 2000 --epochs 2 --output results_smoke  # quick test
"""
import argparse
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from train_cv import TARGET_PRECISION, classification_metrics, format_metrics, precision_threshold


def load_mols(csv_path, max_rows=None):
    """Return (RDKit mols, labels), dropping unparseable SMILES exactly like train_cv.py."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    df = pd.read_csv(csv_path, nrows=max_rows)
    mols = [Chem.MolFromSmiles(s) for s in df["SMILES"]]
    valid = np.array([m is not None for m in mols])
    if not valid.all():
        print(f"Dropping {(~valid).sum()} unparseable SMILES from {csv_path}")
    return [m for m in mols if m is not None], df["label"].to_numpy()[valid].astype(int)


def make_loader(mols, y, args, shuffle):
    from chemprop import data, featurizers

    dps = [data.MoleculeDatapoint(m, np.array([float(t)])) for m, t in zip(mols, y)]
    dset = data.MoleculeDataset(dps, featurizers.SimpleMoleculeMolGraphFeaturizer())
    return data.build_dataloader(dset, batch_size=args.batch_size, num_workers=args.num_workers,
                                 shuffle=shuffle, seed=args.seed)


def build_model(args):
    # Defaults of the `chemprop train` CLI: bond message passing, norm aggregation,
    # batch norm, one-layer FFN. Change these here to match a paper's settings.
    from chemprop import models, nn

    return models.MPNN(
        nn.BondMessagePassing(d_h=args.hidden_size, depth=args.depth, dropout=args.dropout),
        nn.NormAggregation(),
        nn.BinaryClassificationFFN(input_dim=args.hidden_size, hidden_dim=args.ffn_hidden_size,
                                   n_layers=args.ffn_layers, dropout=args.dropout),
        batch_norm=True,
    )


def fit_and_score(args, mols, y, mols_ev, y_ev, ckpt_dir):
    """Train with early stopping on a val slice of (mols, y); score (mols_ev) with the best
    checkpoint. Returns (best checkpoint path, scores on eval, timing info)."""
    import torch
    from lightning import pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    pl.seed_everything(args.seed, workers=True)
    fit_idx, val_idx = train_test_split(np.arange(len(y)), test_size=args.val_frac,
                                        stratify=y, random_state=args.seed)
    train_loader = make_loader([mols[i] for i in fit_idx], y[fit_idx], args, shuffle=True)
    val_loader = make_loader([mols[i] for i in val_idx], y[val_idx], args, shuffle=False)
    eval_loader = make_loader(mols_ev, y_ev, args, shuffle=False)

    shutil.rmtree(ckpt_dir, ignore_errors=True)  # leftovers from an interrupted run
    checkpoint = ModelCheckpoint(dirpath=ckpt_dir, filename="best", monitor="val_loss",
                                 mode="min", save_top_k=1)
    trainer = pl.Trainer(
        logger=False, enable_progress_bar=False, accelerator="auto", devices=1,
        max_epochs=args.epochs,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=args.patience),
                   checkpoint],
    )

    model = build_model(args)
    start = time.perf_counter()
    trainer.fit(model, train_loader, val_loader)
    fit_seconds = time.perf_counter() - start

    # Restore the best-val-loss weights. weights_only=False is safe: this script wrote the file.
    state = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])

    start = time.perf_counter()
    preds = trainer.predict(model, dataloaders=eval_loader)
    scores = torch.cat(preds).float().cpu().numpy().ravel()  # sigmoid probabilities
    predict_seconds = time.perf_counter() - start

    info = {
        "n_train": len(fit_idx),
        "n_val": len(val_idx),
        "n_eval": len(y_ev),
        "epochs_run": trainer.current_epoch,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }
    return checkpoint.best_model_path, scores, info


def main():
    parser = argparse.ArgumentParser(description="K-fold CV + held-out test for Chemprop.")
    parser.add_argument("--split", default="Random", help="Subdirectory of --data_dir (e.g. Random, Cluster)")
    parser.add_argument("--data_dir", default="data/Hitgen_split")
    parser.add_argument("--dataset", default="CK1d_HitGen_binary", help="CSV prefix before _train/_test")
    parser.add_argument("--output", "--output_dir", dest="output_dir", default="results")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.1, help="Early-stopping slice of each training set")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience in epochs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--ffn_hidden_size", type=int, default=300)
    parser.add_argument("--ffn_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int,
                        default=max(0, int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) - 1),
                        help="DataLoader worker processes (featurization runs here)")
    parser.add_argument("--max_rows", type=int, default=None, help="Read only the first N rows (for testing)")
    args = parser.parse_args()

    import torch
    print(f"CUDA available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    total_start = time.perf_counter()
    split_dir = Path(args.data_dir) / args.split
    out_dir = Path(args.output_dir) / args.split / "chemprop"
    fold_dir = out_dir / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    mols, y = load_mols(split_dir / f"{args.dataset}_train.csv", args.max_rows)
    mols_te, yte = load_mols(split_dir / f"{args.dataset}_test.csv", args.max_rows)
    load_seconds = time.perf_counter() - start
    print(f"Loaded train {len(y)}, test {len(yte)} in {load_seconds:.1f}s")

    fold_rows, oof_parts = [], []
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        info_path, oof_path = fold_dir / f"fold{fold}_info.json", fold_dir / f"fold{fold}_oof.csv"
        if info_path.exists() and oof_path.exists():
            print(f"[chemprop] fold {fold}: already done, loading saved scores")
            info, part = json.loads(info_path.read_text()), pd.read_csv(oof_path)
        else:
            _, scores, info = fit_and_score(args, [mols[i] for i in tr_idx], y[tr_idx],
                                            [mols[i] for i in va_idx], y[va_idx],
                                            out_dir / "checkpoints" / f"fold{fold}")
            part = pd.DataFrame({"fold": fold, "train_index": va_idx,
                                 "label": y[va_idx], "score": scores})
            part.to_csv(oof_path, index=False)
            info_path.write_text(json.dumps(info, indent=2))
            print(f"[chemprop] fold {fold}: fit {info['fit_seconds']:.1f}s, "
                  f"{info['epochs_run']} epochs")
        fold_rows.append({"fold": fold, **info})
        oof_parts.append(part)
        pd.DataFrame(fold_rows).to_csv(out_dir / "cv_folds.csv", index=False)
        pd.concat(oof_parts).to_csv(out_dir / "cv_oof_scores.csv", index=False)

    # Hit threshold from the stacked held-out CV scores only (never from test)
    oof = pd.concat(oof_parts)
    threshold = precision_threshold(oof["label"].to_numpy(), oof["score"].to_numpy())
    print(f"[chemprop] threshold for CV precision >= {TARGET_PRECISION}: {threshold:.6f}")

    # Per-fold metrics at that threshold. Optimistic: the threshold was tuned on these scores.
    for row, part in zip(fold_rows, oof_parts):
        row.update(classification_metrics(part["label"].to_numpy(), part["score"].to_numpy(), threshold))
    pd.DataFrame(fold_rows).to_csv(out_dir / "cv_folds.csv", index=False)
    folds = pd.DataFrame(fold_rows).drop(columns="fold")

    # Final model: fit on all of train, call test hits at the fixed CV-derived threshold
    best_ckpt, test_scores, test_info = fit_and_score(args, mols, y, mols_te, yte,
                                                      out_dir / "checkpoints" / "final")
    shutil.copy(best_ckpt, out_dir / "final_model.ckpt")  # usable with `chemprop predict`
    pd.DataFrame({"label": yte, "score": test_scores}).to_csv(out_dir / "test_scores.csv", index=False)
    test_metrics = {**test_info, **classification_metrics(yte, test_scores, threshold)}
    print(f"[chemprop] test: fit {test_metrics['fit_seconds']:.1f}s, {format_metrics(test_metrics)}")

    summary = {
        "model": "chemprop",
        "split": args.split,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "target_precision": TARGET_PRECISION,
        "threshold": threshold,
        "hyperparameters": {k: getattr(args, k) for k in (
            "val_frac", "epochs", "patience", "batch_size", "hidden_size", "depth",
            "ffn_hidden_size", "ffn_layers", "dropout")},
        "cv_mean": folds.mean().to_dict(),
        "cv_std": folds.std().to_dict(),
        "cv_total_fit_seconds": folds["fit_seconds"].sum(),
        "test": test_metrics,
        "load_seconds": load_seconds,
        "total_seconds": time.perf_counter() - total_start,
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
