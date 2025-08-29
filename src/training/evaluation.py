import numpy as np


class FengShuiEvaluator:
    def __init__(self, test_dataset):
        self.test_dataset = test_dataset
        self.metrics = {
            'rmse': self._calculate_rmse,
            'mae': self._calculate_mae,
            'r_squared': self._calculate_r_squared,
        }

    def evaluate_model(self, model, pipeline):
        """全面评估模型性能

        Args:
            model: 待评估模型
            pipeline: 风水分析管道

        Returns:
            Dict: 评估结果
        """
        results = {}
        predictions = []
        ground_truth = []

        for sample in self.test_dataset:
            # 模型预测
            prediction = pipeline.analyze(sample['image_path'])
            predictions.append(prediction['overall_score'])

            # 真实标注
            ground_truth.append(sample['true_score'])

        # 计算各种指标
        for metric_name, metric_func in self.metrics.items():
            results[metric_name] = metric_func(predictions, ground_truth)

        return results

    def _calculate_rmse(self, predictions, ground_truth):
        return np.sqrt(np.mean((np.array(predictions) - np.array(ground_truth)) ** 2))

    def _calculate_mae(self, predictions, ground_truth):
        return np.mean(np.abs(np.array(predictions) - np.array(ground_truth)))

    def _calculate_r_squared(self, predictions, ground_truth):
        correlation_matrix = np.corrcoef(predictions, ground_truth)
        return correlation_matrix[0, 1] ** 2