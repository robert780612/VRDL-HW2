import os
import pandas as pd
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog

from config import cfg
from register_data import svhn_train, get_svhn_metadata



os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

df = pd.read_hdf('train/train_data_processed.h5', 'table')
DatasetCatalog.register("svhn_train", lambda: svhn_train(df))

trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)

trainer.train()

