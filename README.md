!pip install --upgrade opencv-python-headless

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="ghnUQ4hCd5hrI3wnBmKI")
project = rf.workspace("helmetdetection-x8jsy").project("detect-helmet-front")
version = project.version(1)
dataset = version.download("yolov5")


import os
import shutil

%cd /content

!git clone https://github.com/ultralytics/yolov5.git
%cd /content/yolov5
!pip install -r requirements.txt

print("yolov5 디렉토리 내부 파일:")
print(os.listdir('.'))


%cat /content/dataset/With-Helmet-8/data.yaml

%cd /
from glob import glob
import glob

test_images = glob.glob('/content/dataset/With-Helmet-8/test/images/*.jpg')
train_images = glob.glob('/content/dataset/With-Helmet-8/train/images/*.jpg')
valid_images = glob.glob('/content/dataset/With-Helmet-8/valid/images/*.jpg')
# 세 리스트를 합침
img_list = test_images + train_images + valid_images
# 이미지 파일의 총 개수를 출력
print(len(img_list))

from sklearn.model_selection import train_test_split
train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=200)
print(len(train_img_list))
print(len(val_img_list))

with open('/content/dataset/With-Helmet-8/train.txt', 'w') as f:
  f.write('\n'.join(train_img_list) + '\n')

with open('/content/dataset/With-Helmet-8/val.txt', 'w') as f:
  f.write('\n'.join(val_img_list) + '\n')

import yaml

with open('/content/dataset/With-Helmet-8/data.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
print(data)

data['train'] = '/content/dataset/With-Helmet-8/train.txt'
data['val'] = '/content/dataset/With-Helmet-8/val.txt'

with open('/content/dataset/With-Helmet-8/data.yaml', 'w') as f:
  yaml.dump(data, f)
print(data)

%cd /content/yolov5
!python train.py  --img 640 --batch 16 --epochs 100 --data /content/dataset/With-Helmet-8/data.yaml --cfg /content/yolov5/models/yolov5s.yaml --weights yolov5m.pt --name helmet_yolov5m

!python detect.py --weights ./runs/train/helmet_yolov5m/weights/last.pt --conf 1 --source /content/p4.jpg
