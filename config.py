from detectron2.config import get_cfg
from detectron2 import model_zoo

cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("svhn_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_1x.yaml")  # Let training initialize from model zoo

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
cfg.SOLVER.GAMMA = 0.5
cfg.SOLVER.MAX_ITER = 130000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.MOMENTUM = 0.9
# The iteration number to decrease learning rate by GAMMA.
cfg.SOLVER.STEPS = (100000, )


# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.RETINANET.NUM_CLASSES = 10
cfg.MODEL.RETINANET.IOU_THRESHOLDS = [0.4, 0.5]
cfg.MODEL.RETINANET.IOU_LABELS = [0, -1, 1]
# cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.2
# cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 500
cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.5
# cfg.MODEL.RETINANET.IN_FEATURES = ["p3", "p4", "p5"]

# cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[ 4, 8, 16, 32, 64 ]]
# cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
# cfg.INPUT.MIN_SIZE_TRAIN = (32, 64, 128, 256)

cfg.INPUT.RANDOM_FLIP = "none"
