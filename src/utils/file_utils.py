import os
import json


def ensure_directory(directory_path):
    """确保目录存在，如果不存在则创建

    Args:
        directory_path: 目录路径
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def save_results(results, output_path):
    """保存分析结果到文件

    Args:
        results: 分析结果字典
        output_path: 输出文件路径
    """
    ensure_directory(os.path.dirname(output_path))

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)