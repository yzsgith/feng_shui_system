# 风水地理环境评估系统

基于深度学习的风水地理环境评估系统，能够从图像中分析地理环境的风水质量。

## 功能特点

- 多模态特征提取（场景分类、语义分割、目标检测）
- 基于传统风水知识的规则引擎
- 综合风水评分系统
- 可解释性报告生成
- Web API和图形界面支持

## 安装

1. 克隆项目
```bash
git clone <repository-url>
cd feng_shui_system

使用方法
命令行分析
python
复制
下载
from src.main import FengShuiAnalysisPipeline

pipeline = FengShuiAnalysisPipeline()
result = pipeline.analyze("path/to/image.jpg")
print(f"风水评分: {result['overall_score']}")
启动API服务
bash
复制
下载
python scripts/run_api.py
启动图形界面
bash
复制
下载
python scripts/run_gradio.py
批量处理
bash
复制
下载
python scripts/batch_process.py
项目结构
text
复制
下载
feng_shui_system/
├── config/                 # 配置文件
├── src/                   # 源代码
├── scripts/               # 运行脚本
├── models/               # 模型文件
├── data/                 # 数据目录
└── README.md            # 项目说明
许可证
MIT License

text
复制
下载

## 总结

这个完整的代码文件组织实现了风水地理环境评估系统的所有功能模块。系统采用模块化设计，包括：

1. **配置文件**：存储模型参数、风水规则和预处理设置
2. **预处理模块**：处理图像并提取EXIF数据
3. **特征提取模块**：使用多个预训练模型提取多模态特征
4. **知识引擎模块**：应用风水规则计算各维度得分
5. **评分系统模块**：计算综合评分并生成解释性报告
6. **训练模块**：支持模型训练和评估
7. **API模块**：提供Web服务和图形界面
8. **工具函数**：提供可视化和文件操作功能

这种组织方式使得系统具有良好的可维护性和可扩展性，可以方便地添加新的风水规则或替换特征提取模型。