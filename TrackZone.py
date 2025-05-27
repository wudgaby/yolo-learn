import cv2

from ultralytics import solutions
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics.solutions.solutions import SolutionAnnotator

# 中文类别映射
class_map = {
    'person': '人',
    'bicycle': '自行车',
    'car': '汽车',
    'motorcycle': '摩托车',
    'airplane': '飞机',
    'bus': '公交车',
    'train': '火车',
    'truck': '卡车',
    'boat': '船',
    'traffic light': '红绿灯',
    'fire hydrant': '消防栓',
    'stop sign': '停车标志',
    'parking meter': '停车计时器',
    'bench': '长椅',
    'bird': '鸟',
    'cat': '猫',
    'dog': '狗',
    'horse': '马',
    'sheep': '羊',
    'cow': '牛',
    'elephant': '大象',
    'bear': '熊',
    'zebra': '斑马',
    'giraffe': '长颈鹿',
    'backpack': '背包',
    'umbrella': '雨伞',
    'handbag': '手提包',
    'tie': '领带',
    'suitcase': '行李箱',
    'frisbee': '飞盘',
    'skis': '滑雪板',
    'snowboard': '单板滑雪',
    'sports ball': '运动球',
    'kite': '风筝',
    'baseball bat': '棒球棒',
    'baseball glove': '棒球手套',
    'skateboard': '滑板',
    'surfboard': '冲浪板',
    'tennis racket': '网球拍',
    'bottle': '瓶子',
    'wine glass': '酒杯',
    'cup': '杯子',
    'fork': '叉子',
    'knife': '刀',
    'spoon': '勺子',
    'bowl': '碗',
    'banana': '香蕉',
    'apple': '苹果',
    'sandwich': '三明治',
    'orange': '橙子',
    'broccoli': '西兰花',
    'carrot': '胡萝卜',
    'hot dog': '热狗',
    'pizza': '披萨',
    'donut': '甜甜圈',
    'cake': '蛋糕',
    'chair': '椅子',
    'couch': '沙发',
    'potted plant': '盆栽',
    'bed': '床',
    'dining table': '餐桌',
    'toilet': '马桶',
    'tv': '电视',
    'laptop': '笔记本电脑',
    'mouse': '鼠标',
    'remote': '遥控器',
    'keyboard': '键盘',
    'cell phone': '手机',
    'microwave': '微波炉',
    'oven': '烤箱',
    'toaster': '烤面包机',
    'sink': '水槽',
    'refrigerator': '冰箱',
    'book': '书',
    'clock': '钟',
    'vase': '花瓶',
    'scissors': '剪刀',
    'teddy bear': '泰迪熊',
    'hair drier': '吹风机',
    'toothbrush': '牙刷'
}

class_map2 = {
    0: "人",
    1: "自行车",
    2: "汽车",
    3: "摩托车",
    4: "飞机",
    5: "公交车",
    6: "火车",
    7: "卡车",
    8: "船",
    9: "红绿灯",
    10: "消防栓",
    11: "停车标志",
    12: "停车计时器",
    13: "长椅",
    14: "鸟",
    15: "猫",
    16: "狗",
    17: "马",
    18: "羊",
    19: "牛",
    20: "大象",
    21: "熊",
    22: "斑马",
    23: "长颈鹿",
    24: "背包",
    25: "雨伞",
    26: "手提包",
    27: "领带",
    28: "行李箱",
    29: "飞盘",
    30: "滑雪板",
    31: "单板滑雪",
    32: "运动球",
    33: "风筝",
    34: "棒球棒",
    35: "棒球手套",
    36: "滑板",
    37: "冲浪板",
    38: "网球拍",
    39: "瓶子",
    40: "酒杯",
    41: "杯子",
    42: "叉子",
    43: "刀",
    44: "勺子",
    45: "碗",
    46: "香蕉",
    47: "苹果",
    48: "三明治",
    49: "橙子",
    50: "西兰花",
    51: "胡萝卜",
    52: "热狗",
    53: "披萨",
    54: "甜甜圈",
    55: "蛋糕",
    56: "椅子",
    57: "沙发",
    58: "盆栽植物",
    59: "床",
    60: "餐桌",
    61: "马桶",
    62: "电视",
    63: "笔记本电脑",
    64: "鼠标",
    65: "遥控器",
    66: "键盘",
    67: "手机",
    68: "微波炉",
    69: "烤箱",
    70: "烤面包机",
    71: "水槽",
    72: "冰箱",
    73: "书",
    74: "时钟",
    75: "花瓶",
    76: "剪刀",
    77: "泰迪熊",
    78: "吹风机",
    79: "牙刷"
}


# # 保存原始 __init__
# old_init = SolutionAnnotator.__init__
#
# def new_init(self, im, line_width=None, font_size=None, font="simhei.ttf", pil=True, example="abc"):
#     # 强制使用中文字体
#     old_init(self, im, line_width, font_size, font, pil, example)
#     if pil:
#         try:
#             self.font = ImageFont.truetype(font, font_size or 20)
#         except Exception as e:
#             print(f"加载字体失败: {e}，将使用默认字体")
#             self.font = ImageFont.load_default()
#
# # 替换 Annotator 的 __init__
# SolutionAnnotator.__init__ = new_init


def track():
    cap = cv2.VideoCapture("media/renliu2.mp4")
    assert cap.isOpened(), "Error reading video file"

    # Define region points 左上，左下，右上，右下
    region_points = [(0, 0), (0, 750), (750, 0), (750, 750)]

    # Video writer
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter("media/out/trackzone_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Init trackzone (object tracking in zones, not complete frame)
    trackzone = solutions.TrackZone(
        show=False,  # display the output
        region=region_points,  # pass region points
        model="model/yolo11n.pt",  # use any model that Ultralytics support, i.e. YOLOv9, YOLOv10
        # line_width=2,  # adjust the line width for bounding boxes and text display
        show_conf=True,
        show_labels=True,
    )
    # trackzone.names = class_map2
    # trackzone.names[0] = "人"

    # Process video
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or processing is complete.")
            break

        results = trackzone(im0)

        # video_writer.write(im0)  # 写入视频文件
        video_writer.write(results.plot_im)  # 写入视频文件
        # 可选：显示检测结果窗口
        cv2.imshow("Custom Output", im0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()  # destroy all opened windows


if __name__ == '__main__':
    track()
