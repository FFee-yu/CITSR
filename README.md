# CITSR: Cytopathology Image Super-Resolution of Portable Microscope Based on Convolutional Window-integration Transformer
[paper](https://ieeexplore.ieee.org/abstract/document/10819978)  
The implementation of CITSR.
## Model
Toolkits can be found at lib directory.  
Models can be found at model directory.  
We recommend Vscode to avoid dependency problems. 
## Dataset
Baidu Cloud: https://pan.baidu.com/s/13Mnjf2GhI-_NdAPd6yZFfw (password: cyto)  
Google Drive: https://drive.google.com/drive/folders/1BRD5FX01mVLMUVQ3vTEzY1XZ85HV8dS8  
(Please extract the six files, "hr_part1-6", separately, and then place the extracted files into the "hr" folder.)
## Dataset Folder Structure
```
new_data/ 
│
├── lr/
│   ├── (lr image folder 1)
│   ├── ...
│   ├── (lr image folder n)
│
├── hr/
│   ├── (hr image folder 1)
│   ├── ...
│   ├── (hr image folder n)
│
├── train/
│   ├── lr.txt
│   ├── hr.txt
│
├── test/
│   ├── lr.txt
│   ├── hr.txt
```
Note: `lr` and `hr` are the image folders for super-resolution, where `train` and `test` contain the file paths for `lr` and `hr` images read during training and testing, respectively.
## Train
see train.py and for more details. 
## Test
see test.py and for more details.
## Citation
```
@ARTICLE{10819978,
  author={Zhang, Jinyu and Cheng, Shenghua and Liu, Xiuli and Li, Ning and Rao, Gong and Zeng, Shaoqun},
  journal={IEEE Transactions on Computational Imaging}, 
  title={Cytopathology Image Super-Resolution of Portable Microscope Based on Convolutional Window-Integration Transformer}, 
  year={2025},
  volume={11},
  number={},
  pages={77-88},
  keywords={Transformers;Microscopy;Feature extraction;Image reconstruction;Superresolution;Convolutional neural networks;Standards;Convolutional codes;Computer architecture;Computational modeling;Convolutional window-integration;cytopathology image;portable microscope;super-resolution;Transformer},
  doi={10.1109/TCI.2024.3522761}}
```
