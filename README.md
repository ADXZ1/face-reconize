# Face Recognition System (人脸识别系统)

这是一个基于 Python 的人脸识别系统，支持图片识别和实时摄像头识别。

## 功能特点

- 支持图片中的人脸检测和识别
- 支持实时摄像头人脸检测和识别
- 使用 dlib 和 face_recognition 库实现高精度人脸识别
- 支持多人脸同时识别

## 环境要求

- Python 3.6+
- dlib
- face_recognition
- opencv-python
- numpy

## 安装说明

1. 克隆仓库：
```
git clone https://github.com/your-username/face-reconize.git
cd face-reconize
```

2. 安装依赖：
```
pip install -r requirements.txt
```

### 实时摄像头人脸识别

运行以下命令启动实时摄像头识别：
```
python face-recognization-camera.py
```

### 图片中的人脸识别

运行以下命令启动图片中的人脸识别：
```
python face-recognization.py
```

## 文件说明

- `face-recognization.py`: 图片人脸识别程序
- `face-recognization-camera.py`: 实时摄像头人脸识别程序
- `requirements.txt`: 项目依赖文件

## 注意事项

1. 首次运行前请确保已下载所需的模型文件
2. 使用摄像头识别时，请确保系统已正确识别摄像头设备
3. 建议在良好的光照条件下使用，以提高识别准确率

## License

MIT License

## 贡献指南

欢迎提交 Issues 和 Pull Requests 来帮助改进项目。