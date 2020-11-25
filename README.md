# VRDL-HW2
Code for Selected Topics in Visual Recognition using Deep Learning Homework 2

## Briefly Introduction
This repository implements a SVHN digit detector. I used [RetinaNet](https://arxiv.org/abs/1708.02002) with [Resnet50](https://arxiv.org/abs/1512.03385) backbone. It achieves mAP 0.4214 and requires only 50 ms inference time per image.

## Hardware
The following specs were used to create the original solution.
- Ubuntu 16.04 LTS
- Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
- NVIDIA GTX 1080ti

## Reproducing Submission
To reproduct my submission, do the following steps:
1. [Requirement](#requirement)
2. [Dataset Preparation](#dataset-preparation)
3. [Configuration](#configuration)
4. [Training](#training)
5. [Inference](#inference)

## Requirement
The requirements are listed as below
- Python >= 3.6
- numpy
- tqdm
- torch
- torchvision
- tensorboard
- h5py
- pandas 
- matplotlib
- cv2
- pillow
- detectron2

## Dataset Preparation
### Download Official Image
Download and extract SVHN dataset

### Prepare Images
After downloading and converting images, run the following command to convert "digitStruct.mat" file into "train_data_processed.h5" format.
```
python data_preprocess.py
```

## Configuration
Set the configuration in the *config.py* (Model config, Dataset config...). You can find the detail explaination in this [website](https://detectron2.readthedocs.io/modules/config.html#config-references).

## Training
To train models, run following commands. All training log and trained model are saved in "output" directory.
```
$ python train.py
```

## Inference
If trained weights are prepared, you can run the following command to generate json file which contains predicted results.
```
$ python eval.py
```
And if you set ```visualize = False``` to ```True```. It will plot predicted box on the image and save it.
The final detection performance on the SVHN test dataset is mAP 0.42140.
## Reference
- This repository is based on [Detectron2](https://github.com/facebookresearch/detectron2).++
The code for parsing mat file is modified from [pavitrakumar78](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/construct_datasets.py)

