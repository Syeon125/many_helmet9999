{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyOG0zupzvoQZY6iiY7stxKz",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Syeon125/many_helmet9999/blob/main/Untitled2.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wuhcDs5xULJ8",
        "outputId": "bbe891d9-af81-4230-d76e-48bc9f068d3e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[34m\u001b[1mdetect: \u001b[0mweights=['./runs/train/helmet_yolov5m/weights/last.pt'], source=/content/p4.jpg, data=data/coco128.yaml, imgsz=[640, 640], conf_thres=1.0, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_csv=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1\n",
            "YOLOv5 🚀 v7.0-330-gb20fa802 Python-3.10.12 torch-2.3.0+cu121 CUDA:0 (Tesla T4, 15102MiB)\n",
            "\n",
            "Fusing layers... \n",
            "YOLOv5s summary: 157 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs\n",
            "image 1/1 /content/p4.jpg: 640x640 (no detections), 11.5ms\n",
            "Speed: 0.6ms pre-process, 11.5ms inference, 14.1ms NMS per image at shape (1, 3, 640, 640)\n",
            "Results saved to \u001b[1mruns/detect/exp4\u001b[0m\n"
          ]
        }
      ],
      "source": [
        "!pip install --upgrade opencv-python-headless\n",
        "\n",
        "!pip install roboflow\n",
        "\n",
        "from roboflow import Roboflow\n",
        "rf = Roboflow(api_key=\"ghnUQ4hCd5hrI3wnBmKI\")\n",
        "project = rf.workspace(\"helmetdetection-x8jsy\").project(\"detect-helmet-front\")\n",
        "version = project.version(1)\n",
        "dataset = version.download(\"yolov5\")\n",
        "\n",
        "\n",
        "import os\n",
        "import shutil\n",
        "\n",
        "%cd /content\n",
        "\n",
        "!git clone https://github.com/ultralytics/yolov5.git\n",
        "%cd /content/yolov5\n",
        "!pip install -r requirements.txt\n",
        "\n",
        "print(\"yolov5 디렉토리 내부 파일:\")\n",
        "print(os.listdir('.'))\n",
        "\n",
        "\n",
        "%cat /content/dataset/With-Helmet-8/data.yaml\n",
        "\n",
        "%cd /\n",
        "from glob import glob\n",
        "import glob\n",
        "\n",
        "test_images = glob.glob('/content/dataset/With-Helmet-8/test/images/*.jpg')\n",
        "train_images = glob.glob('/content/dataset/With-Helmet-8/train/images/*.jpg')\n",
        "valid_images = glob.glob('/content/dataset/With-Helmet-8/valid/images/*.jpg')\n",
        "# 세 리스트를 합침\n",
        "img_list = test_images + train_images + valid_images\n",
        "# 이미지 파일의 총 개수를 출력\n",
        "print(len(img_list))\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=200)\n",
        "print(len(train_img_list))\n",
        "print(len(val_img_list))\n",
        "\n",
        "with open('/content/dataset/With-Helmet-8/train.txt', 'w') as f:\n",
        "  f.write('\\n'.join(train_img_list) + '\\n')\n",
        "\n",
        "with open('/content/dataset/With-Helmet-8/val.txt', 'w') as f:\n",
        "  f.write('\\n'.join(val_img_list) + '\\n')\n",
        "\n",
        "import yaml\n",
        "\n",
        "with open('/content/dataset/With-Helmet-8/data.yaml', 'r') as f:\n",
        "    data = yaml.load(f, Loader=yaml.FullLoader)\n",
        "print(data)\n",
        "\n",
        "data['train'] = '/content/dataset/With-Helmet-8/train.txt'\n",
        "data['val'] = '/content/dataset/With-Helmet-8/val.txt'\n",
        "\n",
        "with open('/content/dataset/With-Helmet-8/data.yaml', 'w') as f:\n",
        "  yaml.dump(data, f)\n",
        "print(data)\n",
        "\n",
        "%cd /content/yolov5\n",
        "!python train.py  --img 640 --batch 16 --epochs 100 --data /content/dataset/With-Helmet-8/data.yaml --cfg /content/yolov5/models/yolov5s.yaml --weights yolov5m.pt --name helmet_yolov5m\n",
        "\n",
        "!python detect.py --weights ./runs/train/helmet_yolov5m/weights/last.pt --conf 1 --source /content/p4.jpg\n"
      ]
    }
  ]
}