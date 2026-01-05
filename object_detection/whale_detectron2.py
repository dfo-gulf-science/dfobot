# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# assign datasets
from detectron2.data.datasets import register_coco_instances
register_coco_instances("whales_train", {}, "/home/stoyelq/my_hot_storage/dfobot_working/whale/228-noaa-images-20251022.json", "/home/stoyelq/my_hot_storage/dfobot_working/whale/train")
register_coco_instances("whales_test", {}, "/home/stoyelq/my_hot_storage/dfobot_working/whale/228-noaa-images-20251022_val.json", "/home/stoyelq/my_hot_storage/dfobot_working/whale/val")


# visualize:
whales_train_metadata = MetadataCatalog.get("whales_train")
dataset_dicts = DatasetCatalog.get("whales_train")
import random
from detectron2.utils.visualizer import Visualizer
if False:
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=whales_train_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow('image', vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# training:
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("whales_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.OUTPUT_DIR = "./output"
cfg.DEVICE = "cuda:0"
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 5
cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
cfg.SOLVER.MAX_ITER = 500   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()


# validate:

register_coco_instances("whales_test", {}, "/home/stoyelq/my_hot_storage/dfobot_working/whale/228-noaa-images-20251022_val.json", "/home/stoyelq/my_hot_storage/dfobot_working/whale/val")
test_metadata = MetadataCatalog.get("whales_test")

from detectron2.utils.visualizer import ColorMode
import glob

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("whales_test", )
predictor = DefaultPredictor(cfg)

for imageName in glob.glob('/home/stoyelq/my_hot_storage/dfobot_working/whale/val/*JPG'):
    im = cv2.imread(imageName)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=test_metadata,
                   scale=0.8
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("image", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()