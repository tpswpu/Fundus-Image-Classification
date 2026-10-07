import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)

def wilson_ci(p, n, z=1.96):
    if n == 0: return 0, 1
    den = 1 + z ** 2 / n
    center = p + z ** 2 / (2 * n)
    dev = np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)
    return max(0, (center - z * dev) / den), min(1, (center + z * dev) / den)

def calculate_and_save_metrics(y_true, y_pred, y_prob, class_map, output_dir, hw_stats, time_stats):
    labels, names = list(class_map.keys()), list(class_map.values())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = {}

    global_mcc = matthews_corrcoef(y_true, y_pred)
    acc = np.diag(cm).sum() / cm.sum() if cm.sum() > 0 else 0
    acc_ci = wilson_ci(acc, cm.sum())

    try:
        macro_auc = roc_auc_score(label_binarize(y_true, classes=labels), y_prob, multi_class='ovr', average='macro')
    except:
        macro_auc = 0.0

    for i, name in enumerate(names):
        tp, fn = cm[i, i], cm[i, :].sum() - cm[i, i]
        fp, tn = cm[:, i].sum() - cm[i, i], cm.sum() - (cm[:, i].sum() + cm[i, :].sum() - cm[i, i])
        sens, spec = tp / (tp + fn) if tp + fn > 0 else 0, tn / (tn + fp) if tn + fp > 0 else 0
        ppv, npv = tp / (tp + fp) if tp + fp > 0 else 0, tn / (tn + fn) if tn + fn > 0 else 0

        metrics[name] = {
            'Sensitivity': sens, 'Sens_CI_low': wilson_ci(sens, tp + fn)[0], 'Sens_CI_high': wilson_ci(sens, tp + fn)[1],
            'Specificity': spec, 'Spec_CI_low': wilson_ci(spec, tn + fp)[0], 'Spec_CI_high': wilson_ci(spec, tn + fp)[1],
            'PPV': ppv, 'PPV_CI_low': wilson_ci(ppv, tp + fp)[0], 'PPV_CI_high': wilson_ci(ppv, tp + fp)[1],
            'NPV': npv, 'NPV_CI_low': wilson_ci(npv, tn + fn)[0], 'NPV_CI_high': wilson_ci(npv, tn + fn)[1],
            'F1-Score': 2 * (ppv * sens) / (ppv + sens) if ppv + sens > 0 else 0
        }

    df = pd.DataFrame(metrics).T
    df.to_csv(os.path.join(output_dir, "metrics_report.csv"))

    report = f"--- Global Metrics ---\nAccuracy: {acc:.4f} (95% CI: {acc_ci[0]:.4f}-{acc_ci[1]:.4f})\n"
    report += f"Macro-AUC: {macro_auc:.4f}\nMatthews Correlation Coefficient (MCC): {global_mcc:.4f}\n\n"
    report += f"--- Hardware & Efficiency ---\nParameters: {hw_stats['params']}\nFLOPs: {hw_stats['flops']}\n"
    report += f"Total Inference Time: {time_stats['total']:.2f} s\nAverage Latency: {time_stats['avg']:.2f} ms\n\n"
    report += f"--- Per-Class Metrics ---\n{df.to_string()}"

    with open(os.path.join(output_dir, "classification_report.txt"), 'w') as f:
        f.write(report)
    logger.info("Metrics report generated successfully.")
    return df