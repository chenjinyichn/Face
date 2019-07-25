# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time

# "If ..." clause is added by Chen
if __name__ == '__main__':
    detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)

    # 待识别的图片路径
    path = r'C:\Users\User\Desktop\cjy\faceImages'
    names = os.listdir(path)
    for name in names:
        # 建立文件夹以保存提取结果
        result_path = 'C:/Users/User/Desktop/cjy/faceImageGray/' + name
        folder = os.path.exists(result_path)
        if not folder:
            os.makedirs(result_path)

        img_names = os.listdir(os.path.join(path, name))
        for i, img_name in enumerate(img_names):
            img = cv2.imread(os.path.join(path, name, img_name))
            # run detector
            results = detector.detect_face(img)

            if results is not None:
                total_boxes = results[0]
                points = results[1]

                # extract aligned face chips
                chips = detector.extract_image_chips(img, points, 144, 0.37)

                for j, chip in enumerate(chips):
                    gray_img = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
                    # 显示提取后的面部 cv2.imshow(窗口的名字，待显示图像)
                    # cv2.imshow(name + str(i) + str(j),  gray_img)
                    cv2.imwrite(result_path + '/' + str(i) + '_' + str(j) + '.jpg', gray_img)
