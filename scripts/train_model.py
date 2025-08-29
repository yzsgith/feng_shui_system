#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.dataset import FengShuiDataset
from src.training.trainer import FengShuiTrainer, FengShuiRegressionModel
from src.preprocessing.image_preprocessor import FengShuiImagePreprocessor
from src.feature_extraction.multi_modal_extractor import MultiModalFeatureExtractor
from transformers import TrainingArguments
import torch


def main():
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir="./models/trained_feng_shui",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False
    )

    # 初始化预处理器和特征提取器
    preprocessor = FengShuiImagePreprocessor()
    feature_extractor = MultiModalFeatureExtractor()

    # 准备训练数据（这里需要替换为实际数据）
    train_image_paths = []  # 训练图像路径列表
    train_labels = []  # 对应的风水评分列表

    val_image_paths = []  # 验证图像路径列表
    val_labels = []  # 对应的风水评分列表

    train_dataset = FengShuiDataset(train_image_paths, train_labels, preprocessor)
    val_dataset = FengShuiDataset(val_image_paths, val_labels, preprocessor)

    # 创建模型
    regression_model = FengShuiRegressionModel(feature_extractor)

    # 创建训练器并开始训练
    trainer = FengShuiTrainer(regression_model, training_args)
    trainer.train(train_dataset, val_dataset)

    # 保存训练好的模型
    regression_model.save_pretrained("./models/trained_feng_shui/final")


if __name__ == "__main__":
    main()