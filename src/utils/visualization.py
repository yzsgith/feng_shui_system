import matplotlib.pyplot as plt
import numpy as np


class FengShuiVisualizer:
    def __init__(self):
        pass

    def create_radar_chart(self, rule_scores):
        """创建风水规则得分雷达图

        Args:
            rule_scores: 规则得分字典

        Returns:
            matplotlib.figure.Figure: 雷达图
        """
        rules = list(rule_scores.keys())
        scores = [rule_scores[r]['raw_score'] for r in rules]

        angles = np.linspace(0, 2 * np.pi, len(rules), endpoint=False).tolist()
        scores += scores[:1]  # 闭合雷达图
        angles += angles[:1]  # 闭合雷达图

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, scores, 'o-', linewidth=2)
        ax.fill(angles, scores, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), rules)
        ax.set_title('风水维度评分雷达图')
        ax.grid(True)

        return fig

    def create_score_card(self, overall_score, grade):
        """创建风水评分卡片

        Args:
            overall_score: 综合评分
            grade: 风水等级

        Returns:
            matplotlib.figure.Figure: 评分卡片
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.6, f"{overall_score:.1f}",
                fontsize=40, ha='center', va='center')
        ax.text(0.5, 0.3, grade,
                fontsize=15, ha='center', va='center')
        ax.set_title('综合风水评分')
        ax.axis('off')

        return fig