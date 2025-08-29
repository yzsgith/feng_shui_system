import torch.nn as nn
from transformers import (
    AutoImageProcessor, AutoModelForImageClassification,
    AutoModelForSemanticSegmentation, DetrForObjectDetection,
    DetrImageProcessor
)
import yaml
from .custom_layers import CustomFeatureExtractor


class MultiModalFeatureExtractor(nn.Module):
    def __init__(self, model_config_path="config/model_config.yaml"):
        super().__init__()

        # 加载模型配置
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)

        # 场景分类器
        self.scene_processor = AutoImageProcessor.from_pretrained(
            model_config['models']['scene_classifier']['name']
        )
        self.scene_model = AutoModelForImageClassification.from_pretrained(
            model_config['models']['scene_classifier']['name']
        )

        # 语义分割模型
        self.segmentation_processor = AutoImageProcessor.from_pretrained(
            model_config['models']['semantic_segmenter']['name']
        )
        self.segmentation_model = AutoModelForSemanticSegmentation.from_pretrained(
            model_config['models']['semantic_segmenter']['name']
        )

        # 目标检测模型
        self.detection_processor = DetrImageProcessor.from_pretrained(
            model_config['models']['object_detector']['name']
        )
        self.detection_model = DetrForObjectDetection.from_pretrained(
            model_config['models']['object_detector']['name']
        )

        # 自定义特征提取层
        self.custom_extractor = CustomFeatureExtractor(
            model_config['models']['custom_feature_extractor']['layers']
        )

        # 冻结预训练模型参数（可选）
        for param in self.scene_model.parameters():
            param.requires_grad = False
        for param in self.segmentation_model.parameters():
            param.requires_grad = False
        for param in self.detection_model.parameters():
            param.requires_grad = False

    def forward(self, image_tensor):
        """提取多模态特征

        Args:
            image_tensor: 输入图像张量

        Returns:
            Dict: 包含各种特征向量的字典
        """
        features = {}

        # 提取场景特征
        scene_inputs = self.scene_processor(
            images=image_tensor, return_tensors="pt"
        )
        with torch.no_grad():
            scene_outputs = self.scene_model(**scene_inputs)
        features['scene_logits'] = scene_outputs.logits
        features['scene_probs'] = torch.softmax(scene_outputs.logits, dim=-1)

        # 提取分割特征
        segmentation_inputs = self.segmentation_processor(
            images=image_tensor, return_tensors="pt"
        )
        with torch.no_grad():
            segmentation_outputs = self.segmentation_model(**segmentation_inputs)
        features['segmentation_logits'] = segmentation_outputs.logits

        # 提取目标检测特征
        detection_inputs = self.detection_processor(
            images=image_tensor, return_tensors="pt"
        )
        with torch.no_grad():
            detection_outputs = self.detection_model(**detection_inputs)
        features['detection_logits'] = detection_outputs.logits
        features['detection_boxes'] = detection_outputs.pred_boxes

        # 提取自定义风水特征
        custom_features = self.custom_extractor(image_tensor)
        features['custom_features'] = custom_features

        return features