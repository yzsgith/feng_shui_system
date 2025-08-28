import os
import shutil
from typing import List
[f"{name}" for name in ["","",""]]
config=[f"config/{name}" for name in ["feng_shui_knowledge.json","model_config.yaml","preprocess_config.yaml"]]
data=[f"data/{name}" for name in ["raw/","processed/","annotations/"]]
models=[f"models/{name}" for name in ["feature_extractors/","trained_models/","knowledge_base/"]]

src_preprocessing=[f"src/preprocessing/{name}" for name in ["__init__.py","image_preprocessor.py","data_augmentation.py"]]
src_feature_extraction=[f"src/feature_extraction/{name}" for name in ["__init__.py","image_preprocessor.py","custom_layers.py"]]
src_knowledge_engine=[f"src/knowledge_engine/{name}" for name in ["__init__.py","feng_shui_rules.py","knowledge_graph.py"]]
src_scoring=[f"src/scoring/{name}" for name in ["__init__.py","scoring_system.py","explanation_generator.py"]]
src_training=[f"src/training/{name}" for name in ["__init__.py","dataset.py","trainer.py","evaluation.py"]]
src_api=[f"src/api/{name}" for name in ["__init__.py","fastapi_app.py","gradio_interface.py"]]
src_utils=[f"src/utils/{name}" for name in ["__init__.py","visualization.py","file_utils.py"]]
src_main=["src/main.py"]

tests=[f"tests/{name}" for name in ["unit_tests/","integration_tests/"]]
scripts=[f"scripts/{name}" for name in ["train_model.py","run_api.py","batch_process.py"]]
others=["requirements.txt","Dockerfile","README.md"]
def creatFilesOrDirectorys(pathNames: List[str]) -> None:
    """
    根据提供的路径列表创建文件或目录

    参数:
        pathNames: 路径字符串列表，如果路径以分隔符结尾则创建目录，否则创建文件

    注意:
        - 以函数所在位置为起始目录
        - 如果路径已存在，会先删除再重建
        - 支持创建多级嵌套目录
    """
    # 获取函数所在文件的目录作为基准路径
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for path_name in pathNames:
        # 构建完整路径
        full_path = os.path.join(base_dir, path_name)

        # 检查路径是否已存在
        if os.path.exists(full_path):
            try:
                if os.path.isfile(full_path):
                    os.remove(full_path)  # 删除文件
                    print(f"已删除文件: {full_path}")
                else:
                    shutil.rmtree(full_path)  # 删除目录及其内容
                    print(f"已删除目录: {full_path}")
            except Exception as e:
                print(f"删除 {full_path} 时出错: {e}")
                continue  # 跳过此项，继续处理下一个

        # 判断是创建文件还是目录
        # 如果路径以路径分隔符结尾，则创建目录
        if path_name.endswith(os.sep) or path_name.endswith('/') or path_name.endswith('\\'):
            try:
                os.makedirs(full_path, exist_ok=True)
                print(f"已创建目录: {full_path}")
            except Exception as e:
                print(f"创建目录 {full_path} 时出错: {e}")
        else:
            # 创建文件，确保父目录存在
            parent_dir = os.path.dirname(full_path)
            if parent_dir and not os.path.exists(parent_dir):
                try:
                    os.makedirs(parent_dir, exist_ok=True)
                except Exception as e:
                    print(f"创建父目录 {parent_dir} 时出错: {e}")
                    continue

            # 创建文件
            try:
                with open(full_path, 'w') as f:
                    pass  # 创建空文件
                print(f"已创建文件: {full_path}")
            except Exception as e:
                print(f"创建文件 {full_path} 时出错: {e}")


# 示例使用
if __name__ == "__main__":
    # 测试用例

    test_paths =config+data+models+src_preprocessing+src_feature_extraction+src_knowledge_engine+src_scoring+src_training+src_api+src_utils+src_main+tests+scripts+others

    print("开始创建文件和目录...")
    creatFilesOrDirectorys(test_paths)
    print(test_paths)
    print("操作完成!")
