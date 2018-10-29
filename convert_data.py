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

import cv2
import os
import shutil

images = os.listdir('Images')
anns = os.listdir('Annotations')
images.sort()
anns.sort()
print(images[2])
print(anns[2])

count = 0
for j in anns:
    print(j[:-4])
    ann_path = 'Annotations/' + j
    img_path = 'Images/' + j[:-4] + '.jpg'
    out_img_path = 'gray_images/' + str(count) + '.jpg'
    out_img_source_path = 'gray_source_images/' + str(count) + '.jpg'
    out_ann_path = 'anns/' + str(count) + '.txt'

    img = cv2.imread(img_path,0)
    cv2.imwrite(out_img_source_path,img)    

    img = cv2.resize(img, (480, 300), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(out_img_path,img)

    # shutil.copyfile(ann_path,out_ann_path)

    count = count + 1

# cv2.namedWindow("Image")

# cv2.imshow('Image',img)

# cv2.waitKey(0)

# cv2.destroyAllWindows()
