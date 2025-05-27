import cv2
from ultralytics import YOLO


# 使用预训练模型推理测试
# 命令行 yolo predict model=model/yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
def yolo_predict():
    model = YOLO('model/yolo11s.pt')
    # results = model.predict(source='images/', save=True, save_txt=True)
    results = model("images/1.jpg", save=True, save_txt=True)
    results[0].show()

    # path = model.export(format='onnx')


def yolo_pose():
    model = YOLO('model/yolo11n.pt')
    model = YOLO('runs/best2.pt')
    results = model("images/hand", save=True, save_txt=True)
    # results[0].show()
    # for result in results:
    #     xy = result.keypoints.xy  # x and y coordinates
    #     xyn = result.keypoints.xyn  # normalized
    #     kpts = result.keypoints.data  # x, y, visibility (if available)

    # model = YOLO("model/yolo11n-pose.pt")  # load an official model
    # model = YOLO("runs/best2.pt")  # load a custom model
    # Validate the model
    # metrics = model.val()  # no arguments needed, dataset and settings remembered
    # metrics.box.map  # map50-95
    # metrics.box.map50  # map50
    # metrics.box.map75  # map75
    # metrics.box.maps  # a list contains map50-95 of each category
    # metrics.pose.map  # map50-95(P)
    # metrics.pose.map50  # map50(P)
    # metrics.pose.map75  # map75(P)
    # metrics.pse.maps  # a list contains map50-95(P) of each category


def yolo_track():
    model = YOLO('model/yolo11n.pt')
    results = model.track(source="media/video.mp4", imgsz=640, conf=0.3, iou=0.5,
                          show=False, save=True, save_txt=True, stream=True)
    # for r in results:
    #     boxes = r.boxes  # Boxes object for bbox outputs
    #     masks = r.masks  # Masks object for segment masks outputs
    #     probs = r.probs  # Class probabilities for classification outputs

    # 遍历每一帧的检测结果
    for r in results:
        frame = r.orig_img.copy()

        boxes = r.boxes
        if boxes is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()
        ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(xyxy)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            cls_id = int(clss[i])
            conf = confs[i]
            track_id = int(ids[i]) if ids[i] is not None else -1
            label = f"id:{track_id} {model.names[cls_id]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ⬇️ 缩放图像显示
        # scale = 0.3  # 缩放比例（30%）
        # frame_resized = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        # cv2.imshow('Tracking', frame_resized)
        show_image_auto_resize('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def show_image_auto_resize(winname, img, max_width=1280, max_height=720):
    """
    自动缩放图像适配窗口并显示
    - winname: 窗口名称
    - img: 要显示的图像 (BGR 格式)
    - max_width: 显示窗口最大宽度（默认1280）
    - max_height: 显示窗口最大高度（默认720）
    """

    h, w = img.shape[:2]

    # 计算缩放比例（不超过最大宽高）
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)  # 不放大原图

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    # 创建可调窗口
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname, resized)


if __name__ == '__main__':
    # yolo_predict()
    yolo_pose()
    # yolo_track()
