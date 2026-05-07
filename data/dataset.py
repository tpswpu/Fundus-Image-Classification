import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return torch.tensor([]), torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

class EyeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files, self.labels, self.class_names = self._load_data()

    def _load_data(self):
        image_files, labels, class_names_list = [], [], []
        if not os.path.exists(self.data_dir): return [], [], []
        class_dirs = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
        for class_idx, class_dir_name in enumerate(class_dirs):
            class_names_list.append(class_dir_name)
            class_path = os.path.join(self.data_dir, class_dir_name)
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp')):
                    image_files.append(os.path.join(class_dir_name, img_file))
                    labels.append(class_idx)
        return image_files, labels, class_names_list

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform: image = self.transform(image)
            return image, torch.tensor(self.labels[idx], dtype=torch.long), idx
        except Exception:
            return None

def create_dataloader(config, data_dir, batch_size, is_train=True):
    cfg_data, cfg_loader = config['data'], config['dataloader']
    size = (cfg_data['img_size'], cfg_data['img_size']) if isinstance(cfg_data['img_size'], int) else tuple(cfg_data['img_size'])
    transform = T.Compose([
        T.RandomResizedCrop(size) if is_train else T.Resize(size),
        T.RandomHorizontalFlip() if is_train else T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=cfg_data['mean'], std=cfg_data['std'])
    ])
    dataset = EyeDataset(data_dir, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=cfg_loader['num_workers'],
                      pin_memory=cfg_loader['pin_memory'],
                      persistent_workers=cfg_loader.get('persistent_workers', False),
                      drop_last=is_train, collate_fn=custom_collate_fn)
