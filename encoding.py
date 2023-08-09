'''
author:jiguotong
time:2023/08/08
description:对数据库所有图片进行检测，对齐，编码
note:在更换网络结构(backbone)后一定要重新进行人脸编码，运行encoding.py。
'''

import os
from recognition import FaceRecognition

face_Recog = FaceRecognition(encoding=1)
database_dir = 'data/database'
list_dir = os.listdir(database_dir)
image_paths = []
names = []
for name in list_dir:
    image_paths.append(os.path.join(database_dir, name))
    names.append(name.split("_")[0])

face_Recog.encode_face_dataset(image_paths,names)
