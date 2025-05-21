import torch
from ultralytics import YOLO


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 模型验证
def val():
    # 这里有三种训练方式，三种任选其一
    # 第一种：根据yaml文件构建一个新模型进行训练,若对YOLO8网络进行了修改（比如添加了注意力机制）适合选用此种训练方式。但请注意这种训练方式是重头训练(一切参数都要自己训练),训练时间、资源消耗都是十分巨大的
    # model = YOLO('yolo11.yaml')  # build a new model from YAML

    # 第二种：加载一个预训练模型，在此基础之前上对参数进行调整。这种方式是深度学习界最最主流的方式。由于大部分参数已经训练好，我们仅需根据数据集对模型的部分参数进行微调，因此训练时间最短，计算资源消耗最小。
    model = YOLO('model/yolo11s.pt')  # load a pretrained model (recommended for training)

    # 第三种:根据yaml文件构建一个新模型，然后将预训练模型的参数转移到新模型中，然后进行训练，对YOLO8网络进行改进的适合选用此种训练方式，而且训练时间不至于过长
    # model = YOLO('yolo11.yaml').load('yolo11s.pt')  # build from YAML and transfer weights

    # Train the model
    # data参数指定数据集yaml文件(我这里data.yaml与train、val文件夹同目录)
    # epochs指定训练多少轮
    # imgsz指定图片大小
    # batch 每个批次中的图像数量。在训练过程中，数据被分成多个批次进行处理，每个批次包含一定数量的图像
    # device 训练运行的设备cpu,cuda,mps,'cuda:0'明确使用第一个GPU, '0'、'1' 等：指定某个 GPU ID.
    # https://chatgpt.com/share/682bf302-de60-800b-ad38-805750f6cd8b
    # workers: 数据加载时的工作线程数, windows系统下需设置为0，否则会报错
    # name 用于指定本次验证（val）或训练（train）任务的输出文件夹名称

    #print(torch.cuda.is_available())  # True 表示你有 CUDA 可用
    #print(torch.cuda.device_count())  # 有几张 CUDA GPU
    #print(torch.cuda.get_device_name(0))  # 第 0 张 GPU 名称

    results = model.val(data='dataset/data.yaml', name='val_v1', imgsz=(640, 640), save_txt=True, device=device, rect=False)


def val_hand_keypoints():
    model = YOLO('model/yolo11n-pose.pt')  # load a pretrained model (recommended for training)

    # 会自动下载数据集
    results = model.val(data='datasets/hand-keypoints/data.yml', name='val_v1', imgsz=(640, 640), workers=0, batch=32, epochs=100, device=device)


if __name__ == '__main__':
    # val()
    val_hand_keypoints()