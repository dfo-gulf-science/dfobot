
from model.model_utils import ImageFolderCustom, ClassifierModel
import os

from torch.utils.data import Dataset

import pandas as pd
from torchvision import transforms
from torchvision.transforms import v2
import torch

METADATA_CSV_PATH = "/home/stoyelq/my_hot_storage/dfobot_working/dots/dots.csv"
METADATA_COLUMNS = ['is_dot' ]

class IsDotImageFolderCustom(ImageFolderCustom):

    def __init__(self, targ_dir, transform=None):
        super().__init__(targ_dir, transform)
        self.metadata_df = pd.read_csv(METADATA_CSV_PATH)


    def get_metadata(self, path):
        # make sure this function returns the label from the path
        uuid = path.name.split(".")[0]
        metadata_row = self.metadata_df[(self.metadata_df["uuid"] == uuid)].iloc[0]
        out_tensor = torch.tensor(metadata_row[METADATA_COLUMNS].values[0])
        result = metadata_row["is_dot"]
        uuid = metadata_row["uuid"]
        return out_tensor, result, uuid


def get_is_dot_dataloader(batch_size, max_size=None, config_dict=None):
    NUM_WORKERS = config_dict['NUM_WORKERS']
    CROP_SIZE = config_dict['CROP_SIZE']
    VAL_CROP_SIZE = config_dict['VAL_CROP_SIZE']
    IMAGE_FOLDER_DIR = config_dict['IMAGE_FOLDER_DIR']

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomRotation(180),
            transforms.Resize(CROP_SIZE),
            v2.ToDtype(torch.float32, scale=True),

        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(VAL_CROP_SIZE),
            v2.ToDtype(torch.float32, scale=True),
        ]),
    }

    data_dir = IMAGE_FOLDER_DIR
    image_datasets = {x: IsDotImageFolderCustom(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    if max_size is not None:
        image_datasets['train'] = torch.utils.data.Subset(image_datasets["train"], torch.arange(max_size))
        image_datasets['val'] = torch.utils.data.Subset(image_datasets["val"], torch.arange(max_size))

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders, dataset_sizes

