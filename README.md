# VRDL-HW2
Code for Selected Topics in Visual Recognition using Deep Learning Homework 2

## Hardware
The following specs were used to create the original solution.
- Ubuntu 16.04 LTS
- Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
- NVIDIA GTX 1080ti

## Reproducing Submission
To reproduct my submission, do the following steps:
1. [Requirement](#requirement)
2. [Dataset Preparation](#dataset-preparation)
2. [Training](#training)
3. [Inference](#inference)
4. [Make Submission](#make-submission)

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
All required files except images are already in data directory.
If you generate CSV files (duplicate image list, split, leak.. ), original files are overwritten. The contents will be changed, but It's not a problem.

### Download Official Image
Download and extract *training_data*, *testing_data*, and *training_labels.csv*.
If the Kaggle API is installed, run following command.
```
$ kaggle competitions download -c cs-t0828-2020-hw1
```

### Prepare Images
After downloading and converting images, the data directory is structured as:
```
root_dir
  +- training_data
  +- testing_data
  +- training_labels.csv
  +- src
```

## Configuration
Set the configuration in the *config.py* (Model config, Dataset config...)

## Training
To train models, run following commands.
```
$ python train.py
```

The expected training times are:

Model | GPUs | Image size | Training Epochs | Training Time
------------ | ------------- | ------------- | ------------- | -------------
inception-v3 | 1x 1080ti | 224 | 16 | 6 hours

## Inference
If trained weights are prepared, you can create files that contains class probabilities of images.
```
$ python eval.py
```

## Make Submission
Following command will ensemble of all models and make submissions.
```
$ kaggle competitions submit -c cs-t0828-2020-hw1 -f predictions.csv -m "Message"
```

## Reference
This repository is based on [Detectron2](https://github.com/facebookresearch/detectron2)


