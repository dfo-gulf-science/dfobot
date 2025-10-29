
from model.model_utils import ImageFolderCustom, ClassifierModel, AugmentedModel
import os

from torch.utils.data import Dataset

import pandas as pd
from torchvision import transforms
from torchvision.transforms import v2
import torch

METADATA_CSV_PATH = "/home/stoyelq/my_hot_storage/dfobot_working/ages/ages.csv"
METADATA_COLUMNS = ['dummy']


class AgingImageFolderCustom(ImageFolderCustom):

    def __init__(self, targ_dir, transform=None):
        super().__init__(targ_dir, transform)
        self.metadata_df = pd.read_csv(METADATA_CSV_PATH)


    def get_metadata(self, path):
        # make sure this function returns the label from the path
        uuid = path.name.split(".")[0]
        metadata_row = self.metadata_df[(self.metadata_df["uuid"] == uuid)].iloc[0]
        out_metadata_tensor = torch.tensor(metadata_row[METADATA_COLUMNS].values[0])
        result = torch.tensor([float(metadata_row["annuli"])])
        uuid = metadata_row["uuid"]
        return out_metadata_tensor, result, uuid



def get_aging_dataloaders(batch_size, max_size=None, config_dict=None):
    NUM_WORKERS = config_dict['NUM_WORKERS']
    CROP_SIZE = config_dict['CROP_SIZE']
    VAL_CROP_SIZE =  config_dict['VAL_CROP_SIZE']
    IMAGE_FOLDER_DIR = config_dict['IMAGE_FOLDER_DIR']

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomRotation(180),
            transforms.Resize(CROP_SIZE),
            # v2.RandomResizedCrop(size=CROP_SIZE),
            v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(mean=[0.35, 0.39, 0.37], std=[0.1, 0.11, 0.11]),

        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(VAL_CROP_SIZE),
            v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(mean=[0.35, 0.39, 0.37], std=[0.1, 0.11, 0.11]),

        ]),
    }

    data_dir = IMAGE_FOLDER_DIR
    image_datasets = {x: AgingImageFolderCustom(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    if max_size is not None:
        image_datasets['train'] = torch.utils.data.Subset(image_datasets["train"], torch.arange(max_size))
        image_datasets['val'] = torch.utils.data.Subset(image_datasets["val"], torch.arange(max_size))

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders, dataset_sizes



def get_aging_model(device):
    # model_conv = ClassifierModel(1)
    model_conv = AugmentedModel(num_outputs=1, metadata_length=1)
    model_conv.to(device)
    return model_conv