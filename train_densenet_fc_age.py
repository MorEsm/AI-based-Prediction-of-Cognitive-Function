#!/usr/bin/env python
"""
train_densenet_fc_age.py

Train a DenseNet-style CNN with attention blocks (HFAB/ERB) to predict a
continuous target (e.g. age, episodic-memory score) from square functional
connectivity (FC) matrices stored as .mat files.
M. Esmaeili et al. (2026), Brain-Cognitive Gaps in relation to Dopamine and Health-related Factors: Insights from AI-Driven Functional Connectome Predictions, eLife Sciences Publications, Ltd, 2025. https://doi.org/10.7554/eLife.104053.1

Example
-------
python train_densenet_fc_age.py \
    --data-dir /path/to/FC_Movie \
    --labels-csv /path/to/WM_AGE.csv \
    --subject-id-col SubjectID \
    --target-col AGE \
    --runs-per-subject 3 \
    --output-dir ./runs/densenet_age_run1

Requirements: see requirements.txt (TensorFlow >= 2.10, scikit-learn, scipy,
pandas, numpy).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Input, MaxPool2D
from tensorflow.keras.layers import BatchNormalization, ReLU, concatenate, multiply, add

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("densenet_fc")


# Data loading
@dataclass
class Sample:
    filepath: Path
    subject_id: str
    target: float


def extract_subject_id(filepath: Path) -> str:
    """
    Parse a subject identifier out of a .mat filename / path.

    *** ADAPT THIS TO YOUR ACTUAL NAMING CONVENTION. ***
    The default assumes filenames contain a token like 'sub-0012' or
    'subject012'. This is the single most important function to check before
    trusting any results: silent subject/label mismatches are the easiest
    way to get numbers that look plausible but are meaningless.
    """
    match = re.search(r"(sub(?:ject)?-?\d+)", filepath.stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Could not extract a subject ID from '{filepath.name}'. "
            "Update extract_subject_id() to match your file naming scheme."
        )
    return match.group(1).lower()


def build_sample_index(
    data_dir: Path,
    labels_csv: Path,
    subject_id_col: str,
    target_col: str,
) -> list[Sample]:
    """
    Build an explicit, verifiable list of (filepath, subject_id, target)
    triples by joining .mat files to the labels CSV on subject ID.
    """
    df = pd.read_csv(labels_csv)
    if subject_id_col not in df.columns or target_col not in df.columns:
        raise ValueError(
            f"labels CSV must contain columns '{subject_id_col}' and "
            f"'{target_col}'. Found columns: {list(df.columns)}"
        )
    df[subject_id_col] = df[subject_id_col].astype(str).str.lower()
    label_lookup = dict(zip(df[subject_id_col], df[target_col].astype(float)))

    mat_files = sorted(data_dir.glob("**/*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under {data_dir}")

    samples: list[Sample] = []
    skipped = 0
    for fp in mat_files:
        try:
            sid = extract_subject_id(fp)
        except ValueError as exc:
            log.warning("Skipping file: %s", exc)
            skipped += 1
            continue
        if sid not in label_lookup:
            log.warning("No label found for subject '%s' (file: %s) -- skipping.", sid, fp.name)
            skipped += 1
            continue
        samples.append(Sample(filepath=fp, subject_id=sid, target=label_lookup[sid]))

    if skipped:
        log.warning("%d of %d files were skipped due to missing/unmatched labels.", skipped, len(mat_files))
    log.info("Matched %d FC-map files to labels across %d unique subjects.",
              len(samples), len({s.subject_id for s in samples}))
    if not samples:
        raise RuntimeError("No samples matched between .mat files and labels CSV. Check extract_subject_id().")
    return samples


def load_matrices(samples: list[Sample], mat_key: str, matrix_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all FC matrices into a single float32 array, plus targets and group (subject) IDs."""
    n = len(samples)
    data = np.zeros((n, matrix_size, matrix_size), dtype="float32")
    targets = np.zeros(n, dtype="float32")
    groups = np.empty(n, dtype=object)

    for i, s in enumerate(samples):
        mat = scipy.io.loadmat(s.filepath)
        if mat_key not in mat:
            raise KeyError(
                f"Expected key '{mat_key}' not found in {s.filepath.name}. "
                f"Available keys: {[k for k in mat if not k.startswith('__')]}"
            )
        arr = mat[mat_key]
        if arr.shape != (matrix_size, matrix_size):
            raise ValueError(
                f"{s.filepath.name}: expected shape ({matrix_size}, {matrix_size}), got {arr.shape}"
            )
        data[i] = arr
        targets[i] = s.target
        groups[i] = s.subject_id

    data = np.expand_dims(data, axis=-1)  # (N, H, W, 1)
    return data, targets, groups


def group_train_val_split(
    data: np.ndarray, targets: np.ndarray, groups: np.ndarray, val_fraction: float, seed: int
):
    """Subject-grouped split so repeat runs from one subject never straddle train/val."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(data, targets, groups))
    log.info(
        "Train/val split: %d train samples (%d subjects), %d val samples (%d subjects).",
        len(train_idx), len(set(groups[train_idx])),
        len(val_idx), len(set(groups[val_idx])),
    )
    return (data[train_idx], targets[train_idx]), (data[val_idx], targets[val_idx])


# Model
def enhanced_residual_block(x, n_filters: int):
    """ERB: 1x1 -> 3x3 -> 1x1 convs with two residual additions."""
    conv11 = Conv2D(n_filters, 1, kernel_initializer="he_uniform", padding="same", activation="relu")(x)
    y = Conv2D(n_filters, 3, kernel_initializer="he_uniform", padding="same", activation="relu")(conv11)
    y = add([conv11, y])
    y = Conv2D(n_filters, 1, kernel_initializer="he_uniform", padding="same", activation="relu")(y)
    y = add([conv11, y])
    return y


def high_freq_attention_block(x, n_filters: int):
    """HFAB: a conv branch gated by a sigmoid attention map derived from an ERB."""
    branch = Conv2D(n_filters, 3, kernel_initializer="he_uniform", padding="same", activation="relu")(x)
    y = ReLU()(branch)
    y = enhanced_residual_block(y, n_filters)
    y = ReLU()(y)
    y = Conv2D(n_filters, 3, kernel_initializer="he_uniform", padding="same", activation="relu")(y)
    y = tf.keras.activations.sigmoid(y)
    return multiply([y, branch])


def dense_block(x, n_filters: int, n_layers: int, dropout: float):
    for _ in range(n_layers):
        y = Conv2D(n_filters, 3, kernel_initializer="he_uniform", activation="relu", padding="same")(x)
        y = Conv2D(n_filters, 3, kernel_initializer="he_uniform", activation="relu", padding="same")(y)
        if dropout > 0:
            y = Dropout(dropout)(y)
        x = concatenate([y, x])
    return x


def transition_layer(x, n_filters: int, dropout: float):
    n_channels = x.shape[-1] // 2
    x = Conv2D(n_channels, 3, kernel_initializer="he_uniform", activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=5, strides=1, padding="same")(x)
    x = enhanced_residual_block(x, n_filters)
    x = high_freq_attention_block(x, n_filters)
    if dropout > 0:
        x = Dropout(dropout)(x)
    return x


def build_densenet(
    input_shape: tuple[int, int, int],
    n_outputs: int = 1,
    n_filters: int = 16,
    block_depths: tuple[int, ...] = (2, 6, 6, 2),
    dropout: float = 0.2,
) -> Model:
    inputs = Input(input_shape)
    x = BatchNormalization()(inputs)
    x = Conv2D(n_filters, 3, strides=1, kernel_initializer="he_uniform", activation="relu", padding="same")(x)

    for n_layers in block_depths:
        x = high_freq_attention_block(x, n_filters)
        x = dense_block(x, n_filters, n_layers, dropout)
        x = transition_layer(x, n_filters, dropout)

    x = Flatten()(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(n_outputs, activation="linear")(x)
    return Model(inputs, outputs, name="densenet_fc_regressor")


# Training / evaluation
def train(args: argparse.Namespace) -> None:
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    samples = build_sample_index(
        data_dir=Path(args.data_dir),
        labels_csv=Path(args.labels_csv),
        subject_id_col=args.subject_id_col,
        target_col=args.target_col,
    )

    data, targets, groups = load_matrices(samples, mat_key=args.mat_key, matrix_size=args.matrix_size)
    log.info("Loaded data array: %s, target range [%.3f, %.3f]", data.shape, targets.min(), targets.max())

    (train_x, train_y), (val_x, val_y) = group_train_val_split(
        data, targets, groups, val_fraction=args.val_fraction, seed=args.seed
    )

    # Standardize target using train-set statistics only (avoids leakage).
    target_mean, target_std = train_y.mean(), train_y.std() + 1e-8
    train_y_norm = (train_y - target_mean) / target_std
    val_y_norm = (val_y - target_mean) / target_std

    model = build_densenet(
        input_shape=(args.matrix_size, args.matrix_size, 1),
        n_outputs=1,
        n_filters=args.n_filters,
        dropout=args.dropout,
    )
    model.summary(print_fn=log.info)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate) if args.optimizer == "adam" \
        else tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=0.9)
    model.compile(optimizer=optimizer, loss="mean_absolute_error", metrics=["mse"])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.early_stopping_patience,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(5, args.early_stopping_patience // 2), verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_history.csv")),
    ]

    history = model.fit(
        train_x, train_y_norm,
        validation_data=(val_x, val_y_norm),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # Final held-out evaluation, in original target units.
    val_pred_norm = model.predict(val_x, batch_size=args.batch_size).ravel()
    val_pred = val_pred_norm * target_std + target_mean

    mae = mean_absolute_error(val_y, val_pred)
    rmse = mean_squared_error(val_y, val_pred, squared=False)
    r2 = r2_score(val_y, val_pred)
    log.info("Held-out validation -- MAE: %.4f, RMSE: %.4f, R^2: %.4f", mae, rmse, r2)

    metrics = {"val_mae": mae, "val_rmse": rmse, "val_r2": r2,
               "target_mean": float(target_mean), "target_std": float(target_std)}
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    model.save(output_dir / "final_model.keras")
    log.info("Saved final model and metrics to %s", output_dir)


# CLI
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, type=str, help="Directory containing .mat FC-map files (searched recursively).")
    p.add_argument("--labels-csv", required=True, type=str, help="CSV file with subject IDs and target values.")
    p.add_argument("--subject-id-col", default="SubjectID", type=str, help="Column name for subject ID in labels CSV.")
    p.add_argument("--target-col", default="AGE", type=str, help="Column name for the regression target in labels CSV.")
    p.add_argument("--mat-key", default="IMG_temp", type=str, help="Key inside each .mat file holding the FC matrix.")
    p.add_argument("--matrix-size", default=273, type=int, help="Side length of the (square) FC matrix.")
    p.add_argument("--output-dir", default="./run_output", type=str, help="Where to write model checkpoints, logs, metrics.")
    p.add_argument("--val-fraction", default=0.2, type=float, help="Fraction of subjects held out for validation.")
    p.add_argument("--n-filters", default=16, type=int, help="Base channel width for the network.")
    p.add_argument("--dropout", default=0.2, type=float, help="Dropout rate used throughout the network.")
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--epochs", default=400, type=int)
    p.add_argument("--learning-rate", default=1e-3, type=float)
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument("--early-stopping-patience", default=25, type=int)
    p.add_argument("--seed", default=0, type=int)
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    log.info("Starting run with config: %s", vars(args))
    train(args)


if __name__ == "__main__":
    sys.exit(main())
