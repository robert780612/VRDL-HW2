import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob

import pandas as pd
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from config import cfg
from register_data import get_svhn_metadata

# SVHN dataset contains 33,402 trianing images, 13,068 test images
visualize = False
infolder = 'test'
outfolder = 'pred_5_eval'
model_name = 'model_final.pth'
n = len(glob(os.path.join(infolder, '*.png')))
# n = 10

svhn_metadata = get_svhn_metadata()
class_name = svhn_metadata.get('thing_classes')

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_name)  # path to the model we just trained

cfg.TEST.AUG.FLIP = False

predictor = DefaultPredictor(cfg)


if visualize:
    for x in range(1, n+1):
        print(x)
        im = cv2.imread(os.path.join(infolder, '{}.png'.format(x)))
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        fig,ax = plt.subplots(1)
        ax.imshow(im)
        for idx in range(min(len(outputs['instances'].pred_boxes), 4)):  # at most 4 bounding box
            x1, y1, x2, y2 = outputs['instances'].pred_boxes[idx].tensor[0]
            pred_class = outputs['instances'].pred_classes[idx]
            score = outputs['instances'].scores[idx].item()

            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, class_name[pred_class]+'_{:.2f}'.format(score), color='g')
        plt.savefig(os.path.join(outfolder, '{}.png'.format(x)))
        plt.close()


predictions = []
for x in range(1, n+1):
    print(x)
    im = cv2.imread(os.path.join(infolder, '{}.png'.format(x)))
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    # print(outputs)
    bboxes = []
    labels = []
    scores = []
    for idx in range(min(len(outputs['instances'].pred_boxes), 4)):  # at most 4 bounding box
        x1, y1, x2, y2 = outputs['instances'].pred_boxes[idx].tensor[0]
        pred_class = outputs['instances'].pred_classes[idx]
        score = outputs['instances'].scores[idx].item()
        label = int(class_name[pred_class])
        if label == 0:
            label = 10
        # Output json
        bboxes.append([int(y1.item()), int(x1.item()), int(y2.item()), int(x2.item())])
        labels.append(label)
        scores.append(outputs['instances'].scores[idx].item())
    predictions.append({'bbox': bboxes, 'label': labels, 'score': scores})
outdf = pd.DataFrame(predictions)
outdf.to_json('0786039.json', orient='records')

"""
bbox       [[7, 100, 28, 114], [9, 115, 27, 123]]
label                                      [2, 1]
score    [0.9750642180442811, 0.7433308362960811]
"""

