from PIL import Image
import numpy as np
import cv2
import os

path = 'E:/FaceRecognition/faceImageGray'
# new_path = 'E:/FaceRecognition/faceImageGray_new'

data = []
labels = []
dic = {}
Names = os.listdir(path)
for i, Name in enumerate(Names):
    img_names = os.listdir(path + '/' + Name)
    for j, img_name in enumerate(img_names):
        img = cv2.imread(path + '/' + Name + '/' + img_name)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA) # 修改尺寸
        data.append(img)
        labels.append(i)
        dic[i] = Name
print(dic)
# 把list转化成ndarry
data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int32).reshape(-1,1,)

# 保存为npy文件
# np.save('data.npy', data)
# np.save('labels.npy', labels)