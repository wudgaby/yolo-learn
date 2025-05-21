from ultralytics import YOLO


# 使用预训练模型推理测试
# 命令行 yolo predict model=model/yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
def yolo_predict():
    model = YOLO('model/yolo11s.pt')
    # results = model.predict(source='images/', save=True, save_txt=True)
    results = model("images/1.jpg", save=True, save_txt=True)
    results[0].show()

    # path = model.export(format='onnx')


if __name__ == '__main__':
    yolo_predict()
