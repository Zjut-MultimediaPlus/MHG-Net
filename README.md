

# MHG-Net

Estimating Tropical Cyclones with Analysis on Physical Attribution and Physical Correlations Using a Multi-Modal Hybrid Guided Network (JGR Submit)

<!-- PROJECT SHIELDS -->
 
## Catalogue
- [Schema of the project](#Description-of-file-directory)
- [Operation of the project](#Getting-Started-guide)

[//]: # (- [贡献者]&#40;#贡献者&#41;)

[//]: # (  - [如何参与开源项目]&#40;#如何参与开源项目&#41;)

[//]: # (- [版本控制]&#40;#版本控制&#41;)

[//]: # (- [作者]&#40;#作者&#41;)

[//]: # (- [鸣谢]&#40;#鸣谢&#41;)

### Description-of-file-directory

```
filetree 
├── README.md           #Instructions
├── Config.py           #configuration files
├── Dataset.py          #Dataloader
├── MHG_PPNet.py          #Dataloader
├── Train.py & Test.py  #Code for training and testing the model
```

### Getting-started-guide

Please change the path address of all stored data&models in the config.py according to your own storage path.
Test Datasets are stored in：
https://pan.baidu.com/s/18Ta5wu-z4x0bhI3237GRuA  
Extraction code: 1234  
Data in test_156/ corresponds to predict_npy_path156 in Config.py  

Data in test_diff/ corresponds to predict_npy_path_diff in Config.py


### Author

yht (zjut)

email:329769800@qq.comhantingyan@zjut.edu.cn
