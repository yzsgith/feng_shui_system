#!/usr/bin/env python3
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.main import FengShuiAnalysisPipeline
from src.utils.file_utils import save_results


def main():
    # 初始化风水分析管道
    pipeline = FengShuiAnalysisPipeline()

    # 获取要处理的图像路径列表
    image_dir = "path/to/images"
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 批量处理图像
    results = pipeline.batch_analyze(image_paths)

    # 保存结果
    save_results(results, "output/batch_analysis_results.json")

    print(f"处理完成，共分析 {len(results)} 张图像")


if __name__ == "__main__":
    main()