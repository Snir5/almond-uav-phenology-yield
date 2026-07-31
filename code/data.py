import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_SIZE = 1024

def gray_world_wb(img):
    imgf = img.astype(np.float32)
    mean = imgf.reshape(-1,3).mean(axis=0) + 1e-6
    scale = mean.mean() / mean
    out = np.clip(imgf * scale, 0, 255).astype(np.uint8)
    return out

def adaptive_gamma(img, target_v=0.5, clip=(0.7, 1.4)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[...,2].astype(np.float32)/255.0
    v_mean = float(np.clip(v.mean(), 0.05, 0.95))
    gamma = np.log(v_mean) / np.log(max(target_v, 1e-6))
    gamma = float(np.clip(gamma, clip[0], clip[1]))
    x = (img.astype(np.float32)/255.0) ** (1.0/gamma)
    return np.clip(x*255.0,0,255).astype(np.uint8)

def adaptive_clahe(img, base_clip=2.0, tile=(8,8)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[...,2]
    v_std = float(v.std())/255.0
    clip_limit = float(np.clip(base_clip + (0.8 - v_std)*1.0, 1.5, 3.0))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
    hsv[...,2] = clahe.apply(v)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def adaptive_color_deterministic(img):
    wb  = gray_world_wb(img)
    gam = adaptive_gamma(wb, target_v=0.5, clip=(0.8, 1.3))
    out = adaptive_clahe(gam, base_clip=2.0, tile=(8,8))
    return out

def compute_dataset_mean_std(img_dir, sample_size=500):
    img_paths = [p for p in glob.glob(os.path.join(img_dir, "*.jpg"))][:sample_size]
    means, stds = [], []
    for path in img_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        means.append(img.mean(axis=(0, 1)))
        stds.append(img.std(axis=(0, 1)))
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    return tuple(mean), tuple(std)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def adaptive_color_func(x, **kwargs):
    # Named function replacing lambda for multiprocessing compatibility
    return adaptive_color_deterministic(x)

def get_base_transform(norm_type="imagenet", dataset_img_dir=None):
    if norm_type == "dataset" and dataset_img_dir:
        mean, std = compute_dataset_mean_std(dataset_img_dir)
        norm_transform = A.Normalize(mean=mean, std=std)
    elif norm_type == "imagenet":
        norm_transform = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    elif norm_type == "minmax":
        norm_transform = A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    else:
        norm_transform = A.Lambda(image=lambda x, **kwargs: x)
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Equalize(mode='cv', p=0.5),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8,8), p=0.5),
        A.Sharpen(alpha=(0.1,0.3), lightness=(0.7,1.0), p=0.4),
        norm_transform,
        ToTensorV2()
    ])

def get_train_transform(norm_type="imagenet", dataset_img_dir=None):
    base = get_base_transform(norm_type, dataset_img_dir)
    aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.Transpose(p=0.3),
        A.RandomGamma(gamma_limit=(60, 140), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=15, p=0.4),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.6), p=0.3),
    ])
    return A.Compose(aug.transforms + base.transforms)

def get_val_test_transform(norm_type="imagenet", dataset_img_dir=None):
    base = get_base_transform(norm_type, dataset_img_dir)
    return A.Compose([
        A.Lambda(image=adaptive_color_func),
        *base.transforms
    ])

class COCOSegmentationDataset(Dataset):
    def __init__(self, img_dir, ann_path, transform):
        self.img_dir = img_dir
        self.coco = COCO(ann_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.fliplr(image)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))
        mask = np.fliplr(mask)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = (augmented['mask'] > 0).unsqueeze(0).float()
        return image, mask
