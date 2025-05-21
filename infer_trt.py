import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


# 用TensorRT做推理

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def infer(engine, input_data):
    context = engine.create_execution_context()
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(input_data.nbytes)  # 假设输出和输入一样大
    bindings = [int(d_input), int(d_output)]

    cuda.memcpy_htod(d_input, input_data)
    context.execute_v2(bindings)
    output = np.empty_like(input_data)
    cuda.memcpy_dtoh(output, d_output)
    return output

if __name__ == "__main__":
    engine = load_engine("model_int8.engine")
    input_data = np.random.random((1, 3, 224, 224)).astype(np.float32)
    output = infer(engine, input_data)
    print("推理结果：", output)