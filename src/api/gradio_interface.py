import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from ..main import FengShuiAnalysisPipeline


def create_gradio_interface(pipeline):
    def visualize_feng_shui_analysis(image_path):
        """创建风水分析可视化

        Args:
            image_path: 图像文件路径

        Returns:
            Tuple: (可视化图表, 分析摘要)
        """
        result = pipeline.analyze(image_path)

        # 创建可视化图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 显示原始图像
        img = plt.imread(image_path)
        ax1.imshow(img)
        ax1.set_title('原始图像')
        ax1.axis('off')

        # 2. 显示分数雷达图
        rules = list(result['rule_scores'].keys())
        scores = [result['rule_scores'][r]['raw_score'] for r in rules]

        angles = np.linspace(0, 2 * np.pi, len(rules), endpoint=False).tolist()
        scores += scores[:1]  # 闭合雷达图
        angles += angles[:1]  # 闭合雷达图

        ax2 = plt.subplot(222, polar=True)
        ax2.plot(angles, scores, 'o-', linewidth=2)
        ax2.fill(angles, scores, alpha=0.25)
        ax2.set_thetagrids(np.degrees(angles[:-1]), rules)
        ax2.set_title('风水维度评分雷达图')
        ax2.grid(True)

        # 3. 显示整体评分
        ax3.text(0.5, 0.6, f"{result['overall_score']:.1f}",
                 fontsize=40, ha='center', va='center')
        ax3.text(0.5, 0.3, result['grade'],
                 fontsize=15, ha='center', va='center')
        ax3.set_title('综合风水评分')
        ax3.axis('off')

        # 4. 显示主要建议
        recommendations = result['report']['summary']['recommendations'][:3]
        ax4.text(0.1, 0.8, "主要建议:", fontsize=14, weight='bold')
        for i, rec in enumerate(recommendations):
            ax4.text(0.1, 0.6 - i * 0.2, f"• {rec}", fontsize=10,
                     verticalalignment='top')
        ax4.set_title('改善建议')
        ax4.axis('off')

        plt.tight_layout()
        return fig, result['report']['summary']

    # 创建Gradio界面
    demo = gr.Blocks(title="风水地理环境评估系统")

    with demo:
        gr.Markdown("# 🏞️ 风水地理环境评估系统")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath", label="上传地理环境图像")
                analyze_btn = gr.Button("分析风水", variant="primary")

            with gr.Column():
                plot_output = gr.Plot(label="分析结果可视化")
                summary_output = gr.JSON(label="分析摘要")

        analyze_btn.click(
            fn=visualize_feng_shui_analysis,
            inputs=image_input,
            outputs=[plot_output, summary_output]
        )

    return demo