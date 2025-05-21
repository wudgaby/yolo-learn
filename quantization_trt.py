# 量为什么要模型量化？
# 简单说：为了让模型更小、更快、更省资源地部署到各种设备上（尤其是边缘设备、移动端、嵌入式设备等）

# ONNX Runtime 和 TensorRT 都是深度学习模型部署的框架。
# 💻 通用 CPU / GPU 部署	✅ ONNX Runtime
# ⚡ 极致 GPU 性能（NVIDIA GPU）	✅ TensorRT
# 开发初期 / 通用部署： 用 ONNX Runtime，安装快、用法简单。
# 正式部署 / GPU 服务： 把 ONNX 模型转换为 TensorRT Engine，加速推理。
# 边缘设备： 可用 OpenVINO、TensorRT（Jetson）、TFLite 等。



import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_int8_engine(onnx_file_path, engine_file_path, calibration_stream, input_shape):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    # INT8量化
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = MyCalibrator(calibration_stream, input_shape)

    engine = builder.build_engine(network, config)
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
    print(f"INT8 TensorRT engine saved to {engine_file_path}")

# 你需要实现自己的校准器
class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_stream, input_shape):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.stream = calibration_stream
        self.d_input = cuda.mem_alloc(trt.volume(input_shape) * trt.float32.itemsize)
        self.input_shape = input_shape

    def get_batch_size(self):
        return self.input_shape[0]

    def get_batch(self, names):
        try:
            data = next(self.stream)
            cuda.memcpy_htod(self.d_input, data)
            return [int(self.d_input)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        pass

# 示例校准数据生成器
def calibration_data_generator():
    # 这里需要返回numpy数组，形状为input_shape
    # 例如：(1, 3, 224, 224)
    import numpy as np
    for _ in range(10):
        yield np.random.random((1, 3, 224, 224)).astype(np.float32)

if __name__ == "__main__":
    onnx_path = "model.onnx"
    engine_path = "model_int8.engine"
    input_shape = (1, 3, 224, 224)
    build_int8_engine(onnx_path, engine_path, calibration_data_generator(), input_shape)