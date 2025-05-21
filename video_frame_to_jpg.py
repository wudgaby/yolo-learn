import os
from os import mkdir

import cv2

video_path = 'media/video.mp4'
output_path = 'media/output/'
interval = 96  # 每间隔10帧取一张图片

os.makedirs(output_path, exist_ok=True)

if __name__ == '__main__':
    num = 1
    vid = cv2.VideoCapture(video_path)
    while vid.isOpened():
        is_read, frame = vid.read()
        if is_read:
            if num % interval == 1:
                file_name = '%08d' % num
                cv2.imwrite(output_path + str(file_name) + '.jpg', frame)
                cv2.waitKey(1)
            num += 1
        else:
            break
