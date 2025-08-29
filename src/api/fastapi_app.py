from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
from ..main import FengShuiAnalysisPipeline


def create_app(pipeline):
    app = FastAPI(title="风水地理环境评估API")

    @app.post("/analyze-feng-shui")
    async def analyze_feng_shui_endpoint(image: UploadFile = File(...)):
        """分析上传图像的风水

        Args:
            image: 上传的图像文件

        Returns:
            JSONResponse: 包含分析结果的JSON响应
        """
        try:
            # 读取上传的图像
            image_data = await image.read()
            pil_image = Image.open(io.BytesIO(image_data))

            # 临时保存图像
            temp_path = f"/tmp/{image.filename}"
            pil_image.save(temp_path)

            # 进行分析
            result = pipeline.analyze(temp_path)

            # 清理临时文件
            os.remove(temp_path)

            return JSONResponse(content=result)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"分析失败: {str(e)}"
            )

    @app.get("/health")
    async def health_check():
        """健康检查端点

        Returns:
            Dict: 服务状态信息
        """
        return {"status": "healthy", "service": "feng-shui-analysis"}

    return app