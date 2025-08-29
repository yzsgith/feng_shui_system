from transformers import TrainingArguments, Trainer
import torch.nn as nn


class FengShuiRegressionModel(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.regressor = nn.Sequential(
            nn.Linear(128 + 1000 + 150 + 100, 512),  # 调整维度以适应特征
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, pixel_values):
        features = self.feature_extractor(pixel_values)
        # 合并不同来源的特征
        combined_features = torch.cat([
            features['custom_features'],
            features['scene_logits'],
            features['segmentation_logits'].mean(dim=(2, 3)),
            features['detection_logits']
        ], dim=1)

        return self.regressor(combined_features)


class FengShuiTrainer:
    def __init__(self, model, training_args):
        self.model = model
        self.training_args = training_args

    def train(self, train_dataset, eval_dataset):
        """训练风水评分模型

        Args:
            train_dataset: 训练数据集
            eval_dataset: 验证数据集

        Returns:
            Trainer: 训练器对象
        """

        # 定义评估指标
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            rmse = np.sqrt(np.mean((predictions - labels) ** 2))
            return {"rmse": rmse}

        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        # 开始训练
        trainer.train()

        return trainer