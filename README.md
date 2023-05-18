# HandGCAT: Occlusion-Robust 3D Hand Mesh Reconstruction from Monocular Images
This is an official pytorch implementation of the ICME 2023 paper _HandGCAT: Occlusion-Robust 3D Hand Mesh Reconstruction from Monocular Images._
In this repository, we provide PyTorch code for training and testing the proposed HandGCAT on the HO3D and Dexycb dataset.
![image](https://user-images.githubusercontent.com/51118126/229678145-25469907-d52e-4991-9161-3bb12983aa48.png)

# Install
Install PyTorch and Python >= 3.7.4 and run sh requirements.sh.

# Directory
The ${ROOT} is described as below.
```
${ROOT}  
|-- data  
|-- demo
|-- common  
|-- main  
|-- output  
```
data contains data loading codes and soft links to images and annotations directories.
demo contains demo codes.
common contains kernel codes for HandOccNet.
main contains high-level codes for training or testing the network.
output contains log, trained models, visualized outputs, and test result.

# Data
You need to follow directory structure of the data as below.
```
${ROOT}  
|-- data  
|   |-- HO3Dv2
|   |   |-- data
|   |   |   |-- train
|   |   |   |   |-- ABF10
|   |   |   |   |-- ......
|   |   |   |-- evaluation
|   |   |   |-- annotations
|   |   |   |   |-- HO3D_train_data.json
|   |   |   |   |-- HO3D_evaluation_data.json
|   |-- HO3Dv3
|   |   |-- data
|   |   |   |-- train
|   |   |   |   |-- ABF10
|   |   |   |   |-- ......
|   |   |   |-- evaluation
|   |   |   |-- annotations
|   |   |   |   |-- HO3D_train_data.json
|   |   |   |   |-- HO3D_evaluation_data.json
|   |-- DEX_YCB
|   |   |-- data
|   |   |   |-- 20200709-subject-01
|   |   |   |-- ......
|   |   |   |-- annotations
|   |   |   |   |--DEX_YCB_s0_train_data.json
|   |   |   |   |--DEX_YCB_s0_test_data.json
```
# Reference
If this repository is helpful to you, please star it. If you find our work useful in your research, please consider citing:
```
@article{HandGCAT2023,
  title={HandGCAT: Occlusion-Robust 3D Hand Mesh Reconstruction from Monocular Images},
  author={Wang, Shuaibing and Wang, Shunli and Yang, Dingkang and Li, Mingcheng and Qian, Ziyun and Su, Liuzhen and Zhang, Lihua},
  journal={2023 IEEE International Conference on Multimedia & Expo},
  year={2023}
}
```
# Acknowledgements
Some of the code is borrowed from the [HandOccNet](https://github.com/namepllet/HandOccNet) project. We are very grateful for their wonderful implementation.

# Contact
If you have any questions about our work, please contact sbwang21@m.fudan.edu.cn.
