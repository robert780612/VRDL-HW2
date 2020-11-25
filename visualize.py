import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer

from register_data import svhn_train, get_svhn_metadata


def visualize_bounding_box_from_h5(df, indices, folder='visualize'):

    for idx in indices:
        print(idx, len(indices))
        fig,ax = plt.subplots(1)
        ax.imshow(df.loc[idx].img)
        rect = patches.Rectangle((df.loc[idx].left,df.loc[idx].top),df.loc[idx].width,df.loc[idx].height,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        ax.text(0.5, 0.5, df.loc[idx].labels)
        plt.savefig(os.path.join(folder, df.loc[idx].img_name))
        plt.close()


def visualize_bounding_box_from_detetron2(df, folder='detetron2_visualize'):
    dataset_dicts = svhn_train(df)
    svhn_metadata = get_svhn_metadata()
    class_name = svhn_metadata.get('thing_classes')

    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        # visualizer = Visualizer(img[:, :, ::-1], metadata=svhn_metadata, scale=0.5)
        # out = visualizer.draw_dataset_dict(d)
        # plt.imshow(out.get_image()[:, :, ::-1])
        # plt.savefig(os.path.join(folder, d["file_name"]))
        # plt.close()

        fig,ax = plt.subplots(1)
        ax.imshow(img)
        for annotation in d["annotations"]:
            box = annotation['bbox']
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(box[0], box[1], class_name[annotation['category_id']], bbox=dict(boxstyle="square", fc=(1., 0.8, 0.8)))
        plt.savefig(os.path.join(folder, d["file_name"]))
        plt.close()


if __name__=="__main__":
    df = pd.read_hdf('train/train_data_processed.h5', 'table')
    visualize_bounding_box_from_detetron2(df)
    # df = pd.read_hdf('train/train_data_processed.h5', 'table')
    # #indices = (2,5,26,99)
    # indices = list(range(df.shape[0]))
    # visualize_bounding_box(df, indices)

