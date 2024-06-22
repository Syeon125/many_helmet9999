{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyNxd8xUN7k3z5PnSyoARrxJ",
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
      "execution_count": 12,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 147
        },
        "id": "wuhcDs5xULJ8",
        "outputId": "1e871a2d-fd6e-4fda-8312-ca0962d5c748"
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "name 'Image' is not defined",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-12-0eca241f3f36>\u001b[0m in \u001b[0;36m<cell line: 1>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mImage\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mjoin\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'/content/yolov5/runs/train/helmet_yolov5m2/weights/best.pt'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbasename\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mval_img_path\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m: name 'Image' is not defined"
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
        "!python detect.py --weights ./runs/train/helmet_yolov5m2/weights/best.pt --conf 1 --source /content/p4.jpg\n",
        "\n",
        "Image(os.path.join('/content/yolov5/runs/train/helmet_yolov5m2/weights/best.pt', os.path.basename(val_img_path)))\n",
        "\n",
        "\n",
        "\n"
      ]
    }
  ]
}