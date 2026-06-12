import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(cfg_data, is_train: bool):
    height, width = cfg_data.image_size
    transforms = [A.Resize(height=height, width=width)]

    if cfg_data.augmentations.enabled and is_train:
        if cfg_data.augmentations.rotation > 0:
            transforms.append(
                A.Rotate(limit=cfg_data.augmentations.rotation, border_mode=0, p=0.5)
            )
        if cfg_data.augmentations.hflip > 0:
            transforms.append(A.HorizontalFlip(p=cfg_data.augmentations.hflip))
        if cfg_data.augmentations.vflip > 0:
            transforms.append(A.VerticalFlip(p=cfg_data.augmentations.vflip))
        if cfg_data.augmentations.brightness > 0 or cfg_data.augmentations.contrast > 0:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=cfg_data.augmentations.brightness,
                    contrast_limit=cfg_data.augmentations.contrast,
                    p=0.5,
                )
            )
        if cfg_data.augmentations.noise > 0:
            variance = (cfg_data.augmentations.noise * 255.0) ** 2
            transforms.append(A.GaussNoise(var_limit=(0.0, variance), p=0.2))
        if cfg_data.augmentations.blur > 0:
            transforms.append(A.GaussianBlur(blur_limit=(3, 5), p=cfg_data.augmentations.blur))

    transforms.append(A.Normalize(mean=cfg_data.normalize.mean, std=cfg_data.normalize.std))
    transforms.append(ToTensorV2())
    return A.Compose(transforms)
