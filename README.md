

# MHG-Net

Multi-Modal Hybrid Guided Network for Tropical Cyclone Intensity and Size Estimation

<!-- PROJECT SHIELDS -->
 
## Catalogue

- [Schema of the project](#description of file directory)
- [Operation of the project](#Getting started guide)

[//]: # (- [贡献者]&#40;#贡献者&#41;)

[//]: # (  - [如何参与开源项目]&#40;#如何参与开源项目&#41;)

[//]: # (- [版本控制]&#40;#版本控制&#41;)

[//]: # (- [作者]&#40;#作者&#41;)

[//]: # (- [鸣谢]&#40;#鸣谢&#41;)

### 文件目录说明

```
filetree 
├── README.md           #Instructions
├── epoch_375.pth       #Trained model
├── Config.py           #configuration files
├── MHG_Dataset.py      #Dataloader
├── Test_Model.py       #Code for testing the model
```

### 上手指南

Please change the path address of all stored data&models in the config.py according to your own storage path.
Test Datasets are stored in：
https://pan.baidu.com/s/18Ta5wu-z4x0bhI3237GRuA  
Extraction code: 1234  

Data in test_156/ corresponds to predict_npy_path156 in Config.py  

Data in test_diff/ corresponds to predict_npy_path_diff in Config.py


### 作者

yht (zjut)

email:329769800@qq.com
