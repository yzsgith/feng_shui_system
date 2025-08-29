import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import yaml
from .data_augmentation import FengShuiSpecificAugmentation


class FengShuiImagePreprocessor:
    def __init__(self, config_path="config/preprocess_config.yaml"):
        self.config = self._load_config(config_path)
        self.base_transform = self._create_base_transform()
        self.augmentation = FengShuiSpecificAugmentation(self.config['augmentation'])

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _create_base_transform(self):
        return transforms.Compose([
            transforms.Resize(self.config['target_size']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config['normalize_mean'],
                std=self.config['normalize_std']
            )
        ])

    def preprocess(self, image_path, augment=False):
        """预处理图像并提取风水相关特征

        Args:
            image_path: 图像文件路径
            augment: 是否应用数据增强

        Returns:
            Dict: 包含处理后的图像张量、EXIF数据和原始尺寸的字典
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 应用基础变换
        tensor_image = self.base_transform(image)

        # 应用风水特定增强（如果启用）
        if augment:
            tensor_image = self.augmentation(tensor_image)

        # 提取EXIF数据
        exif_data = self._extract_exif_data(image)

        return {
            'tensor': tensor_image.unsqueeze(0),  # 添加批次维度
            'exif': exif_data,
            'original_size': image.size
        }

    def _extract_exif_data(self, image):
        """从图像提取EXIF数据用于方位判断

        Args:
            image: PIL图像对象

        Returns:
            Dict: 包含EXIF数据的字典
        """
        exif_data = {}
        try:
            exif = image._getexif()
            if exif:
                # 提取方向信息
                orientation = exif.get(274, 1)  # 274是方向标签
                exif_data['orientation'] = orientation

                # 提取GPS信息（如果可用）
                gps_info = exif.get(34853, {})
                exif_data['gps'] = gps_info
        except:
            exif_data['orientation'] = 1
            exif_data['gps'] = {}

        return exif_data