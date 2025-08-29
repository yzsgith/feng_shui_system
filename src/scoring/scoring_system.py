import numpy as np


class FengShuiScoringSystem:
    def __init__(self, knowledge_engine):
        self.knowledge_engine = knowledge_engine
        self.rule_weights = knowledge_engine.get_rule_weights()

    def calculate_comprehensive_score(self, rule_scores):
        """计算综合风水评分

        Args:
            rule_scores: 各规则得分字典

        Returns:
            Dict: 包含综合评分和详细信息的字典
        """
        total_score = 0.0
        max_possible = 0.0
        weighted_scores = {}

        for rule_name, score in rule_scores.items():
            weight = self.rule_weights.get(rule_name, 0.1)
            weighted_score = score * weight
            weighted_scores[rule_name] = {
                'raw_score': score,
                'weight': weight,
                'weighted_score': weighted_score
            }
            total_score += weighted_score
            max_possible += weight

        # 归一化到0-100分
        final_score = (total_score / max_possible) * 100 if max_possible > 0 else 0

        return {
            'overall_score': final_score,
            'weighted_scores': weighted_scores,
            'grade': self._get_score_grade(final_score)
        }

    def _get_score_grade(self, score):
        """根据分数获取风水等级

        Args:
            score: 综合风水评分

        Returns:
            str: 风水等级描述
        """
        grading = self.knowledge_engine.knowledge_base['score_grading']

        for grade_name, grade_info in grading.items():
            if grade_info['min'] <= score <= grade_info['max']:
                return f"{grade_info['description']} ({grade_name.replace('_', ' ').title()})"

        return "未知等级"