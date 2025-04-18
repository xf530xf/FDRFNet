# FDRFNet

Paper Title: Full Frequency Decomposition and  Reasoning Fusion for Camouflaged Object Detection



## Usage
### Requirements
* Python 3.8
* Pytorch 1.13
* OpenCV
* Numpy
* Apex
* Timm


### Directory
The directory should be like this:

````
-- src 
-- model (saved model)
-- pre (pretrained model)
-- result (maps)
-- data (train dataset and test dataset)
   |-- TrainDataset
   |   |-- image
   |   |-- mask
   |-- TestDataset
   |   |--NC4K
   |   |   |--image
   |   |   |--mask
   |   |--CAMO
   |   |--COD10K
   ...
   
````



### Train
```
cd src
./train.sh
```
* We implement our method by PyTorch and conduct experiments on 2 NVIDIA RTX4090 GPUs.
* We adopt pre-trained [DeiT](https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth) and [PvT](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth) as backbone networks, which are saved in PRE folder.
* We train our method on 2 backbone settings : ViT and Pvt.
* After training, the trained models will be saved in MODEL folder.

### Test

```
cd src
python test.py
```
* After testing, maps will be saved in RESULT folder


