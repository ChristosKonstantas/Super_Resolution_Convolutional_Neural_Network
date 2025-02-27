**Super Resolution Convolutional Neural Network implementation using DIV2K and Set 14 datasets**


This project is an implementation of [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/pdf/1501.00092) .


You can study the paper provided above and [srcnn_report.pdf](https://github.com/ChristosKonstantas/Super_Resolution_Convolutional_Neural_Network/blob/main/srcnn_report.pdf) file to understand the underlying theoretical aspects.


To execute the project successfully:

* Download and install [cuda toolkit](https://developer.nvidia.com/cuda-toolkit) and the [cuda version](https://pytorch.org/get-started/locally/) that your NVIDIA-GPU supports.
* Download the [Set 14 dataset](https://www.kaggle.com/datasets/ll01dm/set-5-14-super-resolution-dataset) .
* Download the [DIV2K dataset](https://www.kaggle.com/datasets/joe1995/div2k-dataset) .
* Change the paths in main.py, in test.py and in test_scripts.py to your corresponding local directories.
* Execute main.py to train the model.
* Execute test.py to see how good your training performed given a good image and test it versus its bicubic interpolated counterpart.
* Execute test_scripts.py for single image super resolution (SISR).
