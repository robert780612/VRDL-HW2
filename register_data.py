# import pandas as pd


# json.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}])

# df = pd.read_hdf('train/train_data_processed.h5', 'table')
# indices = list(range(df.shape[0]))

# full_data = []
# for id in indices:

#     print(id, len(indices))
#     full_data.append(
#         {'id': id,
#          'file_name': df.loc[id].img_name,
#          'height': df.loc[id].height,
#          'width': df.loc[id].width,
#          'height': df.loc[id].height,
#          }
#     )
    
#     rect = patches.Rectangle((df.loc[idx].left,df.loc[idx].top),df.loc[idx].width,df.loc[idx].height,linewidth=1,edgecolor='r',facecolor='none')
#     ax.add_patch(rect)
#     ax.text(0.5, 0.5, df.loc[idx].labels)
#     plt.savefig(os.path.join(folder, df.loc[idx].img_name))
#     plt.close()


# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import pandas as pd
import torch, torchvision

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode


def svhn_train(df):
    indices = list(range(df.shape[0]))

    img_dir = 'train'
    dataset_dicts = []
    for idx in indices:
        record = {}
        
        filename = os.path.join(img_dir, df.loc[idx].img_name)
        height, width = cv2.imread(filename).shape[:2]
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = df.loc[idx].img_height
        record["width"] = df.loc[idx].img_width
        
        labels = df.loc[idx].labels.split('_')
        left = df.loc[idx].left
        width = df.loc[idx].width
        top = df.loc[idx].top
        height = df.loc[idx].height
        n = df.loc[idx].num_digits
        objs = []
        for i in range(n):
            obj = {
                "bbox": [left + i * width / n, top, left + (i + 1) * width / n, top + height],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(float(labels[i]))-1,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def get_svhn_metadata():
    MetadataCatalog.get("svhn_train").set(thing_classes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'])
    svhn_metadata = MetadataCatalog.get("svhn_train")
    return svhn_metadata


if __name__=="__main__":
    df = pd.read_hdf('train/train_data_processed.h5', 'table')
    dataset_dicts = svhn_train(df)

"""
img_name                                                  1.png
labels                                                  1.0_9.0
top                                                          77
left                                                        246
bottom                                                      300
right                                                       419
width                                                       173
height                                                      223
num_digits                                                    2
img_height                                                  350
img_width                                                   741
img           [[[98, 112, 108], [97, 112, 108], [98, 114, 10...
cut_img       [[[57, 73, 90], [60, 73, 89], [64, 72, 85], [6...
"""


"""
"images": [
    {
        "license": 4,
        "file_name": "000000397133.jpg",
        "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
        "height": 427,
        "width": 640,
        "date_captured": "2013-11-14 17:02:52",
        "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
        "id": 397133
    },
    {
        "license": 1,
        "file_name": "000000037777.jpg",
        "coco_url": "http://images.cocodataset.org/val2017/000000037777.jpg",
        "height": 230,
        "width": 352,
        "date_captured": "2013-11-14 20:55:31",
        "flickr_url": "http://farm9.staticflickr.com/8429/7839199426_f6d48aa585_z.jpg",
        "id": 37777
    },
    ...
]
"""

