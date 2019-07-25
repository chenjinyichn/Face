# coding:utf-8
# from skimage import io, transform
import glob
import os
import tensorflow as tf
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import numpy as np
import time
import cv2
import math
import sys
if __name__ == '__main__':

    # detector = dlib.get_frontal_face_detector()  # 获取人脸分类器
    detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker=4, accurate_landmark=False)
    NAME = {0: 'chenjinyi', 1: 'daiyejun', 2: 'gaohongbin', 3: 'heziwei', 4: 'liangjiahao',
            5: 'luoziying', 6: 'pengchenming', 7: 'tangxiaomin', 8: 'wangjianwei', 9: 'xieyaxue'}

    w = 128
    h = 128
    c = 3

    x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
    y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

    def CNNlayer():
        # 第一个卷积层（128——>64)
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # 第二个卷积层(64->32)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # 第三个卷积层(32->16)
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

        # 第四个卷积层(16->8)
        conv4 = tf.layers.conv2d(
            inputs=pool3,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

        re1 = tf.reshape(pool4, [-1, 8 * 8 * 128])

        # 全连接层
        dense1 = tf.layers.dense(inputs=re1,
                                 units=1024,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        dense2 = tf.layers.dense(inputs=dense1,
                                 units=512,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        logits = tf.layers.dense(inputs=dense2,
                                 units=60,
                                 activation=None,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        return logits
    # ---------------------------网络结束---------------------------

    logits = CNNlayer()
    predict = tf.argmax(logits, 1)

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, 'recog_model/faces.ckpt-5')

    user = input("图片（I）还是摄像头（V）:")
    if user == "I":
        path = input("图片路径是：")
        full_img = cv2.imread(path)
        results = detector.detect_face(full_img)
        if results is not None:
            total_boxes = results[0]
            points = results[1]
            # 画人脸框
            draw = full_img.copy()
            for b in total_boxes:
                cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))
            for p in points:
                for i in range(5):
                    cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)
            cv2.imshow("detection result", draw)
            # extract aligned face chips
            chips = detector.extract_image_chips(full_img, points, 144, 0.37)
            for chip in chips:
                img = cv2.resize(chip, (w, h)) # 调整尺寸
                # 识别
                log = sess.run(logits, feed_dict={x: [img]})
                pre = sess.run(predict, feed_dict={x: [img]})
                l = log[0][pre[0]]
                probability = math.exp(l)/(1 + math.exp(l))
                print('识别结果',NAME[pre[0]])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print('没有识别到人脸')
    else:
        # 打开摄像头
        camera = cv2.VideoCapture(0)
        while(1):
            grab, frame = camera.read()
            # cv2.imshow('frame', frame)
            cv2.imwrite('now.png', frame)
            full_img = cv2.imread("now.png")
            results = detector.detect_face(full_img)
            if results is None:
                continue
            total_boxes = results[0]
            points = results[1]

            chips = detector.extract_image_chips(full_img, points, 144, 0.37)
            img = cv2.resize(chips[0], (w, h))
            log = sess.run(logits, feed_dict={x: [img]})
            pre = sess.run(predict, feed_dict={x: [img]})
            l = log[0][pre[0]]
            probability = math.exp(l) / (1 + math.exp(l))
            print(time.ctime(), NAME[pre[0]])

            # 画人脸框
            draw = frame.copy()
            for b in total_boxes:
                cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))
            for p in points:
                for i in range(5):
                    cv2.circle(draw, (p[i], p[i + 5]), 1, (255, 0, 0), 2)
            cv2.imshow("detection result", draw)
            if cv2.waitKey(1) == 27:
                break
        camera.release()
        cv2.destroyAllWindows()