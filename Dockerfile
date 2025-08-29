FROM python:3.9-slim

WORKDIR /app

# 复制项目文件
COPY . .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建非root用户
RUN useradd -m -u 1000 user
USER user

# 暴露端口
EXPOSE 8000 7860

# 设置默认命令
CMD ["python", "scripts/run_api.py"]