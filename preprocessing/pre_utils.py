import csv
import os
import uuid

import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2
import torch
import cv2

# DATA_DIR = "/home/stoyelq/Documents/dfobot_data/plaice/"
# DATA_DIR = "/home/stoyelq/Documents/dfobot_data/herring/enhanced/"
ORIGINALS_DIR = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/originals/"
RAW_DIR = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/raw/"
RAW_METADATA = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/raw_metadata.csv"
METADATA_DIR = "/home/stoyelq/Documents/dfobot_data/metadata/"
CROP_DIR = "/home/stoyelq/Documents/dfobot_data/cropped_singles/"
# CROP_DIR = "/home/stoyelq/Documents/dfobot_data/cropped_herring_singles/"
TEST_DIR = "/home/stoyelq/Documents/dfobot_data/cropped_singles_test/"
IMAGE_FOLDER_DIR = "/home/stoyelq/Documents/dfobot_data/image_folder/"
IMAGE_FOLDER_DIR = "/home/stoyelq/Documents/dfobot_data/meta_image_folder/"

BUFFER_PX = 5
AREA_THRESHOLD = 0.5
OUT_DIM = (1000, 1000)
TEST_TRAIN_SPLIT = 0.90

def crop_and_save(img, contour, out_dir, buffer=5, outdim=(256, 256)):
    rect = cv2.boundingRect(contour)  # x, y, w, h
    x1 = max(rect[0] - buffer, 0)
    y1 = max(rect[1] - buffer, 0)
    x2 = min((rect[0] + rect[2]) + buffer, img.shape[1])
    y2 = min((rect[1] + rect[3]) + buffer, img.shape[0])
    cropped = img[y1:y2, x1:x2]

    height = y2 - y1
    width = x2 - x1
    scaled_outdim = None
    if width > height:
        scaled_outdim = (outdim[0], int(outdim[1] * height / width))
    else:
        scaled_outdim = (int(outdim[0] * width / height), outdim[1])

    left_pad = outdim[0] - scaled_outdim[0]
    top_pad = outdim[1] - scaled_outdim[1]

    try:
        scaled = cv2.resize(cropped, dsize=scaled_outdim)
        padded = cv2.copyMakeBorder(scaled, top_pad, 0, left_pad, 0, cv2.BORDER_CONSTANT, None, value=0)
        saved = cv2.imwrite(out_dir, padded)
        if not saved:
            print("Could not save {out_dir}".format(out_dir=out_dir))
    except cv2.Error as e:
        print("Error {e}. Could not save {out_dir}".format(e=e, out_dir=out_dir))


def get_data_from_name(img_name, gt_df, herring):
    if herring:
        try:
            sample_id = img_name.split("-")[1]
            fish_number = img_name.split("-")[2].split(".")[0]
        except Exception as e:
            print(img_name)
            raise Exception(e)
        try:
            fish_number = int(fish_number)
        except ValueError:
            fish_number = fish_number
        fish_id = img_name.split(".")[0][5:]
        fish_data_row = gt_df[(gt_df["sample_id"] == int(sample_id)) & (gt_df["fish_number"] == str(fish_number))]
    else:
        fish_id = img_name.split("photo")[0][:-1].split(" ")[0]
        fish_data_row = gt_df[gt_df["specimen_identifier"] == fish_id]
    try:
        fish_age = int(fish_data_row["annulus_count"].iloc[0])
        if fish_age == -99:
            return None, None, None
    except:
        return None, None, None

    length = float(fish_data_row["length_mm"].iloc[0]) / 1000
    weight = float(fish_data_row["weight_g"].iloc[0]) / 1000
    month = float(fish_data_row["collection_date"].iloc[0][6:7]) / 12
    is_male = 1 if fish_data_row["sex"].iloc[0].lower() == "male" else 0
    is_female = 1 if fish_data_row["sex"].iloc[0].lower() == "female" else 0
    is_unknown = 1 if fish_data_row["sex"].iloc[0].lower() == "unknown" else 0
    is_plaice = 0 if herring else 1
    is_herring = 1 if herring else 0
    fish_uuid = uuid.uuid4()
    fish_data = fish_uuid, fish_id, fish_age, length, weight, month, is_male, is_female, is_unknown, is_plaice, is_herring
    return fish_data, fish_age, fish_uuid


def crop_and_isolate():
    # load images
    img_list = os.listdir(ORIGINALS_DIR)
    count = len(img_list)
    # gt_df = load_dmapps_report(herring)
    row_index = 0
    uuid.uuid4()
    with open(RAW_METADATA,'w') as metadata_sheet:
        metadata_sheet_writer = csv.writer(metadata_sheet)
        metadata_sheet_writer.writerow(["uuid", "filename"])

        for img_name in img_list:
        # for img_name in ["T-2008-815-1030(2).jpg"]:
            try:
                count += -1
                if count % 100 == 0:
                    print(count)
                # if count < TEST_TRAIN_SPLIT * len(img_list):
                #     mode = "train"
                # else:
                #     mode = "test"
                img_path = ORIGINALS_DIR + img_name
                img = cv2.imread(img_path)

                # assert img is not None, f"file {img_path} could not be read, check with os.path.exists()"
                if img is None:
                    print(f"file {img_path} could not be read, check with os.path.exists()")
                    continue

                # clip on threshold and convert to grayscale:
                ret, thresh = cv2.threshold(img, 60, 255, 0)
                imgray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

                # Find contours and sort using contour area
                contours = cv2.findContours(imgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                contours = sorted(contours, key=cv2.contourArea, reverse=True)


                # always save largest otolith/contour:
                fish_uuid = uuid.uuid4()
                crop_and_save(img, contours[0], out_dir=f"{RAW_DIR}{fish_uuid}.jpg", buffer=BUFFER_PX, outdim=OUT_DIM)
                metadata_sheet_writer.writerow([fish_uuid, img_name])

                row_index += 1
                # grab second otolith if area is closish:
                if len(contours) > 1:
                    first_area = cv2.contourArea(contours[0])
                    second_area = cv2.contourArea(contours[1])
                    if second_area > AREA_THRESHOLD * first_area:
                        fish_uuid = uuid.uuid4()
                        crop_and_save(img, contours[1], out_dir=f"{RAW_DIR}{fish_uuid}.jpg", buffer=BUFFER_PX, outdim=OUT_DIM)
                        metadata_sheet_writer.writerow([fish_uuid, img_name])
            except Exception as e:
                print(img_name)
                raise Exception(e)


    # csv_name = "metadata_herring.csv" if herring else "metadata_plaice.csv"
    # metadata_df.to_csv(METADATA_DIR + csv_name, index=False)

    # fish_data, fish_age, fish_uuid = get_data_from_name(img_name, gt_df, herring)

    # if fish_age is None:
    #     continue

    # photo_count = int(img_name.split("photo")[1][1]) if "photo" in img_name else 1

    # visualization tool:
    # cv2.drawContours(img, [cnts[1]], -1, (36,255,12), 3)
    # cv2.imshow('image', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


def load_dmapps_report(herring):
    if herring:
        gt_file = os.path.join("/home/stoyelq/Documents/dfobot_data/2019_herring_GT.csv")
    else:
        gt_file = os.path.join("/home/stoyelq/Documents/dfobot_data/2022_RV_GT.csv")
    gt_df = pd.read_csv(gt_file)
    return gt_df
#
# DATA_DIR = "/home/stoyelq/Documents/dfobot_data/plaice/"
# crop_and_isolate(herring=False)
# DATA_DIR = "/home/stoyelq/Documents/dfobot_data/herring/enhanced/"
crop_and_isolate()