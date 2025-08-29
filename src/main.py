import torch
from preprocessing.image_preprocessor import FengShuiImagePreprocessor
from feature_extraction.multi_modal_extractor import MultiModalFeatureExtractor
from knowledge_engine.feng_shui_rules import FengShuiRuleEngine
from scoring.scoring_system import FengShuiScoringSystem
from scoring.explanation_generator import ExplanationGenerator


class FengShuiAnalysisPipeline:
    def __init__(self, config_dir="config"):
        self.preprocessor = FengShuiImagePreprocessor(f"{config_dir}/preprocess_config.yaml")
        self.feature_extractor = MultiModalFeatureExtractor(f"{config_dir}/model_config.yaml")
        self.knowledge_engine = FengShuiRuleEngine(f"{config_dir}/feng_shui_knowledge.json")
        self.scoring_system = FengShuiScoringSystem(self.knowledge_engine)
        self.explanation_generator = ExplanationGenerator(self.knowledge_engine)

    def analyze(self, image_path):
        """完整的风水分析流程

        Args:
            image_path: 图像文件路径

        Returns:
            Dict: 包含完整分析结果的字典
        """
        # 1. 预处理
        preprocessed_data = self.preprocessor.preprocess(image_path)

        # 2. 特征提取
        features = self.feature_extractor(preprocessed_data['tensor'])
        features['exif'] = preprocessed_data['exif']
        features['original_size'] = preprocessed_data['original_size']

        # 3. 应用风水规则
        rule_scores = self.knowledge_engine.apply_rules(features)

        # 4. 计算综合评分
        comprehensive_score = self.scoring_system.calculate_comprehensive_score(rule_scores)

        # 5. 生成解释性报告
        explanation_report = self.explanation_generator.generate_report(
            features, rule_scores, comprehensive_score
        )

        return {
            'image_path': image_path,
            'overall_score': comprehensive_score['overall_score'],
            'grade': comprehensive_score['grade'],
            'rule_scores': comprehensive_score['weighted_scores'],
            'report': explanation_report
        }

    def batch_analyze(self, image_paths):
        """批量分析多张图像

        Args:
            image_paths: 图像路径列表

        Returns:
            List[Dict]: 分析结果列表
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.analyze(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        return results


if __name__ == "__main__":
    # 示例用法
    pipeline = FengShuiAnalysisPipeline()
    result = pipeline.analyze("path/to/your/image.jpg")
    print(f"风水评分: {result['overall_score']}")
    print(f"风水等级: {result['grade']}")