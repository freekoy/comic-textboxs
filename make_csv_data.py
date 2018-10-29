"""
Copyright 2018 freekoy

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import pandas as pd
import random
import cv2

images = os.listdir('gray_images')
anns = os.listdir('anns')
images.sort()
anns.sort()
# print(images[2])
# print(anns[2])

img_data = []
xmin_data = []
xmax_data = []
ymin_data = []
ymax_data = []
class_data = []

# print(len(anns))
random.shuffle(anns)

for j in anns[0:1325]:
    # print(j[:-4])
    ann_path = 'anns/' + j
    img_path = 'gray_source_images/' + j[:-4] + '.jpg'
    frame = j[:-4] + '.jpg'
    img = cv2.imread(img_path)
    print(img_path)
    print(img.shape)
    width = (img.shape)[1]
    height = (img.shape)[0]

    ann = open(ann_path)
    ann = ann.readlines()

    for i in ann:
        # print(i)
        text_box = (i.split('\r'))[0].split(' ')
        
        xmin = int(text_box[1]) / width * 480
        ymin = int(text_box[2]) / height * 300
        xmax = int(text_box[3]) / width * 480
        ymax = int(text_box[4]) / height * 300
        xmin = int(xmin)
        xmax = int(xmax)
        ymin = int(ymin)
        ymax = int(ymax)
        class_id = int(text_box[0])

        img_data.append(frame)
        xmin_data.append(xmin)
        xmax_data.append(xmax)
        ymin_data.append(ymin)
        ymax_data.append(ymax)
        class_data.append(1)
    

# #字典中的key值即为csv中列名
train = pd.DataFrame({'frame':img_data,'xmin':xmin_data,'xmax':xmax_data,'ymin':ymin_data,'ymax':ymax_data,'class_id':class_data})

# # 将DataFrame存储为csv,index表示是否显示行名，default=True
train.to_csv("train.csv",index=False,sep=',')

img_data = []
xmin_data = []
xmax_data = []
ymin_data = []
ymax_data = []
class_data = []

for j in anns[1325:]:
    # print(j[:-4])
    ann_path = 'anns/' + j
    img_path = 'gray_source_images/' + j[:-4] + '.jpg'
    frame = j[:-4] + '.jpg'
    img = cv2.imread(img_path)
    print(img_path)
    print(img.shape)
    width = (img.shape)[1]
    height = (img.shape)[0]

    ann = open(ann_path)
    ann = ann.readlines()

    for i in ann:
        # print(i)
        text_box = (i.split('\r'))[0].split(' ')
        
        xmin = int(text_box[1]) / width * 480
        ymin = int(text_box[2]) / height * 300
        xmax = int(text_box[3]) / width * 480
        ymax = int(text_box[4]) / height * 300
        xmin = int(xmin)
        xmax = int(xmax)
        ymin = int(ymin)
        ymax = int(ymax)
        class_id = int(text_box[0])

        img_data.append(frame)
        xmin_data.append(xmin)
        xmax_data.append(xmax)
        ymin_data.append(ymin)
        ymax_data.append(ymax)
        class_data.append(int(1))

# #字典中的key值即为csv中列名
val = pd.DataFrame({'frame':img_data,'xmin':xmin_data,'xmax':xmax_data,'ymin':ymin_data,'ymax':ymax_data,'class_id':class_data})

# # 将DataFrame存储为csv,index表示是否显示行名，default=True
val.to_csv("val.csv",index=False,sep=',')