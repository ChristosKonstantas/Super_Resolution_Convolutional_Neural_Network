0) Create your own Pycharm project (or in Visual Studio) and install the packages defined above on each .py file. For pytorch better install cuda toolkit of your corresponding GPU cuda support and then install torch
and torch vision from this link: https://pytorch.org/get-started/locally/.
All the other libraries are easy to install from your IDE terminal.
1) First download the Set 14 dataset: https://www.kaggle.com/datasets/ll01dm/set-5-14-super-resolution-dataset
2) Then download the DIV2K dataset: https://www.kaggle.com/datasets/joe1995/div2k-dataset
3) Change the directories in main.py, in test.py and in test_scripts.py to your corresponding directories of your personal computer.
4) Run main.py for training.
5) Run test.py to see how good your training was given a good image but testing it on its bicubic interpolated counterpart.
6) Run test_scripts.py for single image enhancement.
