import os
import logging
import pandas as pd
import torch
from torch import optim
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
from tqdm import tqdm

from models.swin_entropy import SwinEntropyClassifier
from data.dataset import create_dataloader

logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class TrainingEngine:
    def __init__(self, config):
        self.config, self.cfg_train = config, config['train']
        self.device = self.cfg_train['device']

        params = config['model'].get('swin_entropy_classifier_params', {})
        params['head'] = {'num_classes': len(config['predict']['class_names'])}
        params['pretrained'] = config['model'].get('pretrained', True)

        self.model = SwinEntropyClassifier(params).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg_train['optimizer']['lr'],
                                     weight_decay=self.cfg_train['optimizer']['weight_decay'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.scaler = GradScaler(enabled=(self.device == 'cuda' and self.cfg_train['use_amp']))

        self.save_dir = os.path.join(self.cfg_train['save_dir'], config['model']['name'])
        os.makedirs(os.path.join(self.save_dir, 'weight'), exist_ok=True)

        self.train_loader = create_dataloader(config, os.path.join(self.cfg_train['data_root'], "Train"), self.cfg_train['batch_size'], True)
        self.val_loader = create_dataloader(config, os.path.join(self.cfg_train['data_root'], "Val"), self.cfg_train['batch_size'] * 2, False)

        self.start_epoch, self.best_val_acc, self.metrics_log = 0, 0.0, []
        self.early_stopping = EarlyStopping(patience=self.cfg_train['early_stopping']['patience'],
                                            min_delta=self.cfg_train['early_stopping']['min_delta'])

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg_train['epochs']} [Train]")
        for images, labels, _ in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            with autocast(device_type=self.device, enabled=(self.device == 'cuda' and self.cfg_train['use_amp'])):
                preds = self.model(images)
                loss = F.cross_entropy(preds, labels, label_smoothing=self.cfg_train['label_smoothing'])
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item() * images.size(0)
            total_correct += (preds.argmax(1) == labels).sum().item()
            total_samples += images.size(0)
            pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}", acc=f"{total_correct / total_samples:.4f}")
        return total_loss / total_samples, total_correct / total_samples

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        for images, labels, _ in tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{self.cfg_train['epochs']} [Val]"):
            images, labels = images.to(self.device), labels.to(self.device)
            with autocast(device_type=self.device, enabled=(self.device == 'cuda' and self.cfg_train['use_amp'])):
                preds = self.model(images)
                loss = F.cross_entropy(preds, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds.argmax(1) == labels).sum().item()
            total_samples += images.size(0)
        return total_loss / total_samples, total_correct / total_samples

    def train(self):
        logger.info(f"Starting training for {self.config['model']['name']}...")
        for epoch in range(self.start_epoch, self.cfg_train['epochs']):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            self.scheduler.step()

            self.metrics_log.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})
            pd.DataFrame(self.metrics_log).to_csv(os.path.join(self.save_dir, 'training_log.csv'), index=False)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({'model_state_dict': self.model.state_dict()}, os.path.join(self.save_dir, 'weight/best_model.pth'))

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop: 
                logger.info("Early stopping triggered.")
                break