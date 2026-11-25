# shared cv splits, threshold rule, and metric helpers
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_SPLITS = 5
MIN_SPECIFICITY = 0.70
F_BETA = 2.0


# make the same 5 stratified folds (balanced by cycle and target) for all models
def make_cv_splits(df, n_splits=N_SPLITS, seed=SEED):
    strata = df["cycle"].astype(str) + "_" + df["target"].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(df, strata))


# compute all metrics for a given probability threshold
def compute_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "f2": round(float(fbeta_score(y_true, y_pred, beta=F_BETA, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(specificity), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


# pick the threshold with highest F2 while keeping specificity >= 0.70
def find_best_threshold(y_true, y_prob, min_specificity=MIN_SPECIFICITY, beta=F_BETA):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    best_thr = None
    best_f2 = -1.0
    for thr in np.linspace(0.01, 0.99, 981):
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if spec < min_specificity:
            continue
        f2 = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_thr = float(thr)

    if best_thr is None:
        raise ValueError("no threshold reached the minimum specificity")

    return best_thr, compute_metrics(y_true, y_prob, best_thr)


# print a short readable metric summary
def print_metrics(title, metrics):
    print(f"[{title}]")
    print(
        f"  ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | "
        f"F2: {metrics['f2']:.4f} | Thr: {metrics['threshold']:.4f}"
    )
    print(
        f"  Recall: {metrics['recall']:.4f} | Specificity: {metrics['specificity']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}"
    )
