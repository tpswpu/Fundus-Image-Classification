import torch
import torch.nn.functional as F

class SwinGradCAM:
    def __init__(self, model):
        self.model = model
        self.target_layer = model.cam_target
        self.gradients, self.activations = None, None
        self.target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o.detach()))
        self.target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def __call__(self, x, class_idx):
        self.model.eval()
        self.gradients, self.activations = None, None
        with torch.enable_grad():
            output = self.model(x)
            self.model.zero_grad()
            output[:, class_idx].sum().backward(retain_graph=False)

            if self.gradients is None or self.activations is None: return None

            weights = torch.mean(self.gradients, dim=[1, 2], keepdim=True)
            cam = torch.sum(weights * self.activations, dim=3)

            B = x.size(0)
            cam = cam.unsqueeze(1)
            cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            cam = F.relu(cam).squeeze(1)

            for b in range(B):
                cam[b] -= cam[b].min()
                cam[b] /= (cam[b].max() + 1e-8)
            return cam.cpu().numpy()

def occlude_by_cam(img_t, cam, top_ratio=0.2, fill_value=0.0):
    x_occ = img_t.clone()
    cam_t = torch.from_numpy(cam).to(img_t.device)
    threshold = torch.quantile(cam_t.flatten(), 1.0 - top_ratio)
    mask = cam_t >= threshold
    x_occ[:, :, mask] = fill_value
    return x_occ
