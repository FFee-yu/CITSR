# CITSR: Cytopathology Image Super-Resolution of Portable Microscope Based on Convolutional Window-integration Transformer
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
new_data
|---lr
    |---(image folder)
    ...
    |---(image folder)
|---hr
    |---(image folder)
    ...
    |---(image folder)
|---train
    |---lr.txt
    |---hr.txt
|---test
    |---lr.txt
    |---hr.txt
Note: `lr` and `hr` are the image folders for super-resolution, where `train` and `test` contain the file paths for `lr` and `hr` images read during training and testing, respectively.
## Train
see train.py and for more details. 
## Test
see test.py and for more details.
## Citation
