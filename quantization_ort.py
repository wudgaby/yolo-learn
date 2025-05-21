import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

# 1. 定义校准数据读取器
class MyDataReader(CalibrationDataReader):
    def __init__(self, calibration_images):
        self.data = calibration_images
        self.enum_data = iter(self.data)

    def get_next(self):
        try:
            return {"input": next(self.enum_data)}
        except StopIteration:
            return None

# 2. 准备校准数据（假设 input shape 为 (1, 3, 224, 224)）
import numpy as np
calibration_data = [np.random.rand(1, 3, 224, 224).astype(np.float32) for _ in range(10)]
data_reader = MyDataReader(calibration_data)

# 3. 量化
quantize_static(
    model_input="model.onnx",
    model_output="model_int8.onnx",
    calibration_data_reader=data_reader,
    quant_format=None,  # QOperator 格式，适合部署
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    optimize_model=True
)

print("量化完成，已保存为 model_int8.onnx")