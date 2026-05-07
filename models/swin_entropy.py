import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models.feature_extraction import create_feature_extractor

class SwinEntropyClassifier(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.num_classes = model_config['head']['num_classes']
        weights = Swin_T_Weights.DEFAULT if model_config.get('pretrained', True) else None
        base_model = swin_t(weights=weights)

        self.backbone = create_feature_extractor(base_model, return_nodes={'features.7': 'stage4_feat'})
        self.cam_target = nn.Identity()

        hidden_dim = model_config['head'].get('hidden_dim', 512)
        dropout = model_config['head'].get('dropout', 0.5)

        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, hidden_dim),
            nn.ReLU(inplace=False),  
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes)
        )
        for m in self.modules():
            if hasattr(m, 'inplace'): m.inplace = False

    def forward(self, x):
        features = self.backbone(x)
        stage4_feat = self.cam_target(features['stage4_feat'])
        x = stage4_feat.mean(dim=[1, 2])
        return self.classifier(x)
