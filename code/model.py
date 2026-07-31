import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        TP = (inputs * targets).sum(dim=(1, 2, 3))
        FP = ((1 - targets) * inputs).sum(dim=(1, 2, 3))
        FN = (targets * (1 - inputs)).sum(dim=(1, 2, 3))
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        return focal_tversky.mean()

def build_model(device):
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    return model

def dice_coef(pred, target, eps=1e-7):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return ((2. * intersection + eps) / (union + eps)).mean()

def iou_score(pred, target, eps=1e-7):
    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5
    intersection = (pred & target).float().sum((1,2,3))
    union = (pred | target).float().sum((1,2,3))
    return ((intersection + eps) / (union + eps)).mean()
