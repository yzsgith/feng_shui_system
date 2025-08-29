class ExplanationGenerator:
    def __init__(self, knowledge_engine):
        self.knowledge_engine = knowledge_engine

    def generate_report(self, features, rule_scores, comprehensive_score):
        """生成详细的可解释性报告

        Args:
            features: 特征字典
            rule_scores: 各规则得分
            comprehensive_score: 综合评分结果

        Returns:
            Dict: 包含详细分析的报告
        """
        report = {
            'summary': {
                'overall_score': comprehensive_score['overall_score'],
                'grade': comprehensive_score['grade'],
                'strengths': [],
                'weaknesses': [],
                'recommendations': []
            },
            'detailed_analysis': {}
        }

        # 为每个规则生成详细分析
        for rule_name, score_info in comprehensive_score['weighted_scores'].items():
            rule_analysis = self._analyze_rule(rule_name, score_info, features)
            report['detailed_analysis'][rule_name] = rule_analysis

            # 收集优势和劣势
            if score_info['raw_score'] >= 0.7:
                report['summary']['strengths'].append(rule_analysis['strength'])
            elif score_info['raw_score'] <= 0.4:
                report['summary']['weaknesses'].append(rule_analysis['weakness'])

            # 收集建议
            report['summary']['recommendations'].extend(rule_analysis['recommendations'])

        return report

    def _analyze_rule(self, rule_name, score_info, features):
        """分析单个规则的评分结果

        Args:
            rule_name: 规则名称
            score_info: 得分信息
            features: 特征字典

        Returns:
            Dict: 规则分析结果
        """
        analysis = {
            'score': score_info['raw_score'],
            'weight': score_info['weight'],
            'contribution': score_info['weighted_score'],
            'strength': "",
            'weakness': "",
            'recommendations': []
        }

        # 根据不同的规则生成特定的分析
        if rule_name == "mountain_water_balance":
            if score_info['raw_score'] >= 0.8:
                analysis['strength'] = "山水平衡极佳，符合风水宝地标准"
            elif score_info['raw_score'] <= 0.4:
                analysis['weakness'] = "山水元素缺失或失衡"
                analysis['recommendations'].append("考虑增加水体或假山装饰来改善平衡")

        # 其他规则的分析...

        return analysis