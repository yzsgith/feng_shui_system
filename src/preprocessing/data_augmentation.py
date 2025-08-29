import torch
from torchvision import transforms


class FengShuiSpecificAugmentation:
    def __init__(self, augmentation_config):
        self.augmentation = transforms.Compose([
            transforms.ColorJitter(
                brightness=augmentation_config['brightness'],
                contrast=augmentation_config['contrast'],
                saturation=augmentation_config['saturation'],
                hue=augmentation_config['hue']
            ),
            transforms.RandomAdjustSharpness(
                sharpness_factor=augmentation_config['sharpness'],
                p=0.3
            ),
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=augmentation_config['blur_sigma']
            ),
        ])

    def __call__(self, image_tensor):
        return self.augmentation(image_tensor)