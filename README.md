# Super Resolution Convolutional Neural Network


This project is an implementation of [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/pdf/1501.00092) .


You can study the paper provided above and [srcnn_report.pdf](https://github.com/ChristosKonstantas/Super_Resolution_Convolutional_Neural_Network/blob/main/srcnn_report.pdf) file to understand the underlying theoretical concepts.

## Prerequisites
* If you haven't setup cuda, you can download and install [cuda toolkit](https://developer.nvidia.com/cuda-toolkit) and the [cuda version](https://pytorch.org/get-started/locally/) that your NVIDIA-GPU supports.
* Download the [Set 14 dataset](https://www.kaggle.com/datasets/ll01dm/set-5-14-super-resolution-dataset) .
* Download the [DIV2K dataset](https://www.kaggle.com/datasets/joe1995/div2k-dataset) .

## Instructions
```bash
git clone https://github.com/ChristosKonstantas/Super_Resolution_Convolutional_Neural_Network.git
```

### Windows
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
### Unix/Linux/Mac
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### Training
In `src` folder
```bash
python train.py --train_dir "\path\to\DIV2K_train_HR" --eval_dir "\path\to\DIV2K_valid_HR" --output_dir "\path\to\output_dir"
```
### Evaluation

#### Execute scripts

```bash
python infer.py --model_path "..\pretrained\model.pth" --image_path "your\path\to\pngFile"
```
to see how good your training performed given a good image and compare it versus its bicubic interpolated counterpart.

or

```bash
python sisr.py --train_dir "\path\to\DIV2K_train_HR" --eval_dir "\path\to\DIV2K_valid_HR" --first_image "D:\Datasets\DIV2K_train_HR\DIV2K_train_HR\0001.png"
```
for Single Image Super Resolution (SISR).
