from ultralytics import YOLO

# 导出模型为ONNX格式

# 加载预训练模型
model = YOLO('./runs/detect/train_v1015/weights/best.pt')

# 导出模型为ONNX格式
model.export(format='onnx', name='yolov11s', ops=11)