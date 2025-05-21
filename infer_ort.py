import onnxruntime as ort
import numpy as np
import cv2

# 1. 加载模型
session = ort.InferenceSession("model_int8.onnx", providers=['CPUExecutionProvider'])

# 2. 读取图片并预处理
img = cv2.imread("bus.jpg")  # 替换为你的图片路径
img = cv2.resize(img, (224, 224))  # 根据模型输入尺寸调整
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0  # 归一化
img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
img = np.expand_dims(img, axis=0)   # 增加 batch 维度

# 3. 获取输入名
input_name = session.get_inputs()[0].name

# 4. 推理
outputs = session.run(None, {input_name: img})

# 5. 输出结果
print("推理结果：", outputs)