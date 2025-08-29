#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.main import FengShuiAnalysisPipeline
from src.api.gradio_interface import create_gradio_interface


def main():
    # 初始化风水分析管道
    pipeline = FengShuiAnalysisPipeline()

    # 创建Gradio界面
    demo = create_gradio_interface(pipeline)

    # 启动服务
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()