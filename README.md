**Super Resolution Convolutional Neural Network implementation using DIV2K and Set 14 datasets**


This project is the implementation of the paper [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/pdf/1501.00092) .


You can study the paper provided above and [srcnn_report.pdf](https://github.com/ChristosKonstantas/Super_Resolution_Convolutional_Neural_Network/blob/main/srcnn_report.pdf) file to understand the underlying theoretical aspects.

To execute the project successfully:

0) Create your own Pycharm project (or in Visual Studio) and install the packages defined above on each .py file. For pytorch better install cuda toolkit of your corresponding GPU cuda support and then install torch
and torch vision from this [link](https://pytorch.org/get-started/locally/).
All the other libraries are easy to install from your IDE terminal after creating your own virtual environment executing 'python3 -m venv venv'.
1) First download the [Set 14 dataset](https://www.kaggle.com/datasets/ll01dm/set-5-14-super-resolution-dataset)
2) Then download the [DIV2K dataset](https://www.kaggle.com/datasets/joe1995/div2k-dataset)
3) Change the directories in main.py, in test.py and in test_scripts.py to your local directories of preference.
4) Run main.py for training.
5) Run test.py to see how good your training was given a good image but testing it on its bicubic interpolated counterpart.
6) Run test_scripts.py for single image enhancement.
