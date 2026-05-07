import os
import copy
import time
import logging
from collections import defaultdict, OrderedDict
import torch
import numpy as np
from torch.amp import autocast
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

from models.swin_entropy import SwinEntropyClassifier
from utils.cam import occlude_by_cam
from utils.metrics import calculate_and_save_metrics
from utils.visualizer import ResultVisualizer

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

logger = logging.getLogger(__name__)

class EvaluatorEngine:
    def __init__(self, config):
        self.config = config
        self.device = config['predict']['device']
        self.class_map = config['predict']['class_names']
        self.model = self._load_model()
        self.vis = ResultVisualizer(config, self.model)

    def _load_model(self):
        model_params = self.config['model'].get('swin_entropy_classifier_params', {})
        model_params['head'] = {'num_classes': len(self.class_map)}
        model_params['pretrained'] = self.config['model'].get('pretrained', True)

        model = SwinEntropyClassifier(model_params).to(self.device)
        checkpoint = torch.load(self.config['predict']['weight_path'], map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))

        clean_state_dict = OrderedDict()
        for k, v in state_dict.items():
            clean_state_dict[k.replace('module.', '')] = v

        model.load_state_dict(clean_state_dict, strict=False)
        return model.eval()

    def _calculate_flops(self, input_size):
        if not THOP_AVAILABLE: return "N/A", "N/A"
        try:
            temp_model = copy.deepcopy(self.model).cpu().eval()
            flops, params = profile(temp_model, inputs=(torch.randn(1, *input_size).cpu(),), verbose=False)
            return f"{flops / 1e9:.2f} G", f"{params / 1e6:.2f} M"
        except Exception:
            return "N/A", "N/A"

    def evaluate(self):
        cfg_pred = self.config['predict']
        flops_str, params_str = self._calculate_flops((3, self.config['data']['img_size'], self.config['data']['img_size']))
        hardware_stats = {'params': params_str, 'flops': flops_str}

        transform = T.Compose([
            T.Resize((self.config['data']['img_size'], self.config['data']['img_size'])),
            T.CenterCrop((self.config['data']['img_size'], self.config['data']['img_size'])),
            T.ToTensor(),
            T.Normalize(mean=self.config['data']['mean'], std=self.config['data']['std'])
        ])

        in_dir = cfg_pred['input_dir']
        name2idx = {v: k for k, v in self.class_map.items()}

        image_files = []
        for cls_name in sorted(os.listdir(in_dir)):
            if os.path.isdir(os.path.join(in_dir, cls_name)):
                for fname in os.listdir(os.path.join(in_dir, cls_name)):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                        image_files.append((os.path.join(in_dir, cls_name, fname), cls_name))

        if not image_files:
            logger.error("No valid images found.")
            return

        y_true, y_pred, y_prob, txt_res = [], [], [], []
        cam_cnt = defaultdict(int)
        total_latency_ms = 0.0
        cam_opts = cfg_pred.get('visualization', {}).get('grad_cam_options', {})

        start_wall_time = time.perf_counter()
        for img_path, true_cls in tqdm(image_files, desc="Predicting Swin"):
            if true_cls not in name2idx: continue
            try:
                pil_img = Image.open(img_path).convert('RGB')
                img_t = transform(pil_img).unsqueeze(0).to(self.device)
                orig_img = np.array(pil_img)
            except Exception:
                continue

            t_start = time.perf_counter()
            with torch.no_grad():
                with autocast(device_type=self.device, enabled=cfg_pred['use_amp']):
                    out = self.model(img_t)
            if self.device == 'cuda': torch.cuda.synchronize()
            total_latency_ms += (time.perf_counter() - t_start) * 1000

            probs = torch.softmax(out, dim=1).squeeze().cpu().numpy()
            pred_idx = int(np.argmax(probs))
            true_idx = name2idx[true_cls]

            # Occlude experiment
            if self.vis.cam_gen is not None:
                cam = self.vis.cam_gen(img_t, true_idx)
                if cam is not None:
                    for ratio in [0.1, 0.2, 0.3]:
                        img_occ = occlude_by_cam(img_t, cam[0], top_ratio=ratio)
                        with torch.no_grad():
                            with autocast(device_type=self.device, enabled=cfg_pred['use_amp']):
                                out_occ = self.model(img_occ)
                        probs_occ = torch.softmax(out_occ, dim=1).squeeze().cpu().numpy()
                        pred_occ = int(np.argmax(probs_occ))
                        txt_res.append(
                            f"{os.path.basename(img_path)} CAM_OCC_{int(ratio * 100)}% true={true_cls} "
                            f"orig_pred={self.class_map[pred_idx]} occ_pred={self.class_map[pred_occ]} "
                            f"p_true_orig={probs[true_idx]:.4f} p_true_occ={probs_occ[true_idx]:.4f} drop={probs[true_idx] - probs_occ[true_idx]:.4f}"
                        )

            y_true.append(true_idx)
            y_pred.append(pred_idx)
            y_prob.append(probs)
            txt_res.append(f"{os.path.basename(img_path)} {true_cls} {self.class_map[pred_idx]}")

            if cam_opts.get('enabled') and cam_opts.get('mode') == 'first_n' and cam_cnt[true_cls] < cam_opts.get('n_per_class', 5):
                self.vis.save_cam(orig_img, img_t, pred_idx, f"{true_cls}_{os.path.splitext(os.path.basename(img_path))[0]}")
                cam_cnt[true_cls] += 1

        if not y_true: return

        time_stats = {'total': time.perf_counter() - start_wall_time, 'avg': total_latency_ms / len(y_true)}
        
        if cfg_pred.get('save_predictions_to_txt', {}).get('enabled', True):
            with open(os.path.join(self.vis.out_dir, cfg_pred.get('save_predictions_to_txt', {}).get('filename', 'prediction_results.txt')), 'w') as f:
                f.write('\n'.join(txt_res))

        metrics_df = calculate_and_save_metrics(y_true, y_pred, np.array(y_prob), self.class_map, self.vis.out_dir, hardware_stats, time_stats)
        self.vis.plot_confusion_matrix(y_true, y_pred)
        self.vis.plot_roc_curves(y_true, np.array(y_prob))
        self.vis.plot_calibration_curve(y_true, np.array(y_prob))
        self.vis.plot_metrics_barchart(metrics_df)
        logger.info("Evaluation completed successfully!")
