import json
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

from data_loading import DataSplits

CONTEXT_SIZE = 10_000
BATCH = 10_000


def run_foundation_model(splits: DataSplits, seed: int) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_ctx, y_ctx = splits.X_train, splits.y_train
    if len(X_ctx) > CONTEXT_SIZE:
        X_ctx, _, y_ctx, _ = train_test_split(
            X_ctx, y_ctx, train_size=CONTEXT_SIZE, stratify=y_ctx, random_state=seed
        )

    clf = TabPFNClassifier(device=device, random_state=seed)

    t0 = perf_counter()
    clf.fit(X_ctx, y_ctx)
    fit_time = perf_counter() - t0

    X_test, y_test = splits.X_test, splits.y_test
    t0 = perf_counter()
    proba = []
    for i in range(0, len(X_test), BATCH):
        chunk = X_test.iloc[i:i + BATCH]
        proba.extend(clf.predict_proba(chunk))
    proba = np.array(proba)
    pred_time = perf_counter() - t0

    proba = proba / proba.sum(axis=1, keepdims=True)
    y_pred = clf.classes_[np.argmax(proba, axis=1)]

    result = {
        "auroc": float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=clf.classes_)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "context_size": int(len(X_ctx)),
        "n_test": int(len(y_test)),
        "fit_sec": float(fit_time),
        "predict_sec": float(pred_time),
        "elapsed_sec": float(fit_time + pred_time),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "seed": int(seed),
    }
    with open(f"foundation_covertype_seed{seed}.json", "w") as f:
        json.dump(result, f, indent=2)
    return result
