import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from utils.cam import SwinGradCAM

def configure_plotting_style():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.dpi'] = 300

def save_source_data(data, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(data).to_csv(os.path.join(output_dir, filename), index=False)

class ResultVisualizer:
    def __init__(self, config, model):
        self.config = config
        self.class_names = config['predict']['class_names']
        self.out_dir = os.path.join(config['predict']['visualization']['output_base_dir'], config['model']['name'])
        self.src_dir = os.path.join(self.out_dir, "source_data")
        self.cam_dir = os.path.join(self.out_dir, "grad_cam_maps")

        for d in [self.out_dir, self.src_dir, self.cam_dir]: os.makedirs(d, exist_ok=True)
        configure_plotting_style()
        self.cam_gen = SwinGradCAM(model) if config['predict']['visualization'].get('grad_cam_options', {}).get('enabled', False) else None

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=list(self.class_names.keys()))
        cm_norm = cm / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        names = list(self.class_names.values())
        save_source_data(pd.DataFrame(cm, index=names, columns=names), self.src_dir, "confusion_matrix.csv")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax, xticklabels=names, yticklabels=names)
        ax.set_ylabel('True Class'); ax.set_xlabel('Predicted Class')
        fig.savefig(os.path.join(self.out_dir, "confusion_matrix.png"), bbox_inches='tight')
        plt.close(fig)

    def plot_roc_curves(self, y_true, y_prob):
        y_true_bin = label_binarize(y_true, classes=list(self.class_names.keys()))
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, color in zip(range(len(self.class_names)), cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])):
            if y_true_bin.shape[1] > i:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                ax.plot(fpr, tpr, color=color, lw=2, label=f"{self.class_names[i]} (AUC = {auc(fpr, tpr):.3f})")
                save_source_data({'fpr': fpr, 'tpr': tpr}, self.src_dir, f"roc_{self.class_names[i]}.csv")
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right")
        fig.savefig(os.path.join(self.out_dir, "roc_curves.png"), bbox_inches='tight')
        plt.close(fig)

    def plot_calibration_curve(self, y_true, y_prob):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        for i, name in self.class_names.items():
            if i < y_prob.shape[1]:
                frac, mean_pred = calibration_curve(np.array(y_true) == i, y_prob[:, i], n_bins=10)
                ax.plot(mean_pred, frac, "s-", label=name)
                save_source_data({'mean_predicted_value': mean_pred, 'fraction_of_positives': frac}, self.src_dir, f"calib_{name}.csv")
        ax.set_ylabel('Fraction of Positives'); ax.set_xlabel('Mean Predicted Value')
        ax.set_ylim([-0.05, 1.05])
        ax.legend(loc="lower right")
        fig.savefig(os.path.join(self.out_dir, "calibration_curves.png"), bbox_inches='tight')
        plt.close(fig)

    def plot_metrics_barchart(self, metrics_df):
        m_plot = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1-Score']
        df_plot = metrics_df.loc[:, m_plot].T
        ci_low, ci_high = df_plot.copy(), df_plot.copy()
        for m in m_plot:
            prefix = 'Sens' if m == 'Sensitivity' else 'Spec' if m == 'Specificity' else m
            low_key, high_key = f'{prefix}_CI_low', f'{prefix}_CI_high'
            if low_key in metrics_df.index and high_key in metrics_df.index:
                ci_low[m], ci_high[m] = metrics_df.loc[low_key], metrics_df.loc[high_key]
        errs = np.stack([df_plot.values - ci_low.values, ci_high.values - df_plot.values]).transpose(2, 0, 1)

        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot.plot(kind='bar', ax=ax, yerr=errs, error_kw=dict(ecolor='black', lw=1, capsize=3))
        ax.set_ylabel('Score'); ax.set_xlabel('Class')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        fig.savefig(os.path.join(self.out_dir, "metrics_barchart.png"), bbox_inches='tight')
        plt.close(fig)

    def save_cam(self, orig_img, img_t, pred_idx, prefix):
        if not self.cam_gen: return
        heatmap = self.cam_gen(img_t, pred_idx)
        if heatmap is not None:
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap[0]), cv2.COLORMAP_JET)
            heatmap_resized = cv2.resize(heatmap_colored, (orig_img.shape[1], orig_img.shape[0]))
            alpha = self.config['predict']['visualization']['grad_cam_options'].get('grad_cam_alpha', 0.5)
            overlay = cv2.addWeighted(heatmap_resized, alpha, cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR), 1 - alpha, 0)
            cv2.imwrite(os.path.join(self.cam_dir, f"{prefix}_cam.png"), overlay)