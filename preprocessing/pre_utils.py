import csv
import os
import shutil
import uuid

import numpy as np
import pandas as pd
import shapely.wkt
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

DFODOTS_METADATA = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/dfo_dots_ages.csv"
ORACLE_METADATA = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/yellowtail_oracle_dump.csv"

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


def crop_by_ccords(img, x1, y1, x2, y2, out_dir, outdim=(256, 256)):
    cropped = img[y1:y2, x1:x2]
    saved = cv2.imwrite(out_dir, cropped)


def crop_and_isolate():
    # load images
    img_list = os.listdir(ORIGINALS_DIR)
    count = len(img_list)
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


def import_crab():
    CRAB_ORIGINALS_DIR = "/home/stoyelq/Documents/crab"
    CRAB_METADATA = "/home/stoyelq/my_hot_storage/dfobot/crab/originals.csv"
    OUT_DIR = "/home/stoyelq/my_hot_storage/dfobot/crab/raw/"
    # load images
    year_list = os.listdir(CRAB_ORIGINALS_DIR)
    with open(CRAB_METADATA,'w') as metadata_sheet:
        metadata_sheet_writer = csv.writer(metadata_sheet)
        metadata_sheet_writer.writerow(["uuid", "year", "filename"])
        for year in year_list:
            img_list = os.listdir(os.path.join(CRAB_ORIGINALS_DIR, year))
            count = len(img_list)
            for img_name in img_list:
            # for img_name in ["T-2008-815-1030(2).jpg"]:
                try:
                    count += -1
                    if count % 100 == 0:
                        print(count)
                    img_path = os.path.join(CRAB_ORIGINALS_DIR, year,  img_name)
                    img = cv2.imread(img_path)

                    # assert img is not None, f"file {img_path} could not be read, check with os.path.exists()"
                    if img is None:
                        print(f"file {img_path} could not be read, check with os.path.exists()")
                        continue

                    scaled = cv2.resize(img, dsize=(500, 500))
                    new_uuid = uuid.uuid4()

                    out_path = os.path.join(OUT_DIR, f"{new_uuid}.jpg")
                    saved = cv2.imwrite(out_path, scaled)
                    metadata_sheet_writer.writerow([new_uuid, year, img_name])

                except Exception as e:
                    print(img_name)
                    raise Exception(e)


def crop_and_transform(img_path, coords, out_path):
    outdim = OUT_DIM
    # clip on threshold and convert to grayscale:
    img = cv2.imread(img_path)
    ret, thresh = cv2.threshold(img, 60, 255, 0)
    imgray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    buffer = BUFFER_PX
    # Find contours and sort using contour area
    contours = cv2.findContours(imgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # grab largest contour containing center point:
    center_x, center_y, edge_x, edge_y = coords
    for contour in contours:
        # save largest contour containing center:
        rect = cv2.boundingRect(contour)  # x, y, w, h
        x1 = max(rect[0] - buffer, 0)
        y1 = max(rect[1] - buffer, 0)
        x2 = min((rect[0] + rect[2]) + buffer, img.shape[1])
        y2 = min((rect[1] + rect[3]) + buffer, img.shape[0])
        if x1 < center_x < x2 and y1 < center_y < y2:
            cropped = img[y1:y2, x1:x2]
            height = y2 - y1
            width = x2 - x1

            # step 1, shift points by crop:
            center_x = center_x - x1
            center_y = center_y - y1
            edge_x = edge_x - x1
            edge_y = edge_y - y1

            scaled_outdim = None
            if width > height:
                scaled_outdim = (outdim[0], int(outdim[1] * height / width))
                scale_factor = outdim[0] / width
                center_x = center_x  * scale_factor
                center_y = center_y  * scale_factor
                edge_x = edge_x  * scale_factor
                edge_y = edge_y  * scale_factor

            else:
                scaled_outdim = (int(outdim[0] * width / height), outdim[1])
                scale_factor = outdim[1] / height
                center_x = center_x  * scale_factor
                center_y = center_y  * scale_factor
                edge_x = edge_x  * scale_factor
                edge_y = edge_y  * scale_factor

            left_pad = outdim[0] - scaled_outdim[0]
            top_pad = outdim[1] - scaled_outdim[1]

            center_x = center_x + left_pad
            center_y = center_y + top_pad
            edge_x = edge_x + left_pad
            edge_y = edge_y + top_pad

            scaled = cv2.resize(cropped, dsize=scaled_outdim)
            padded = cv2.copyMakeBorder(scaled, top_pad, 0, left_pad, 0, cv2.BORDER_CONSTANT, None, value=0)
            saved = cv2.imwrite(out_path, padded)
            return max(0, int(center_x)), max(0, int(center_y)), max(0, int(edge_x)), max(0, int(edge_y))

def train_val_splitter(in_dir, out_dir, split=0.9):
    img_list = os.listdir(in_dir)
    os.makedirs(f"{out_dir}train", exist_ok=True)
    os.makedirs(f"{out_dir}val", exist_ok=True)

    count = 0
    for img_name in img_list:
        count += 1
        src = os.path.join(in_dir, img_name)
        if count < split * len(img_list):
            mode = "train"
        else:
            mode = "val"
        dst = os.path.join(out_dir, mode, img_name)
        shutil.copy(src, dst)



def row_ager_and_writer(row, ages_writer, oracle_df):
    year = int(row["filename"].split("-")[1])
    mission_number = int(row["filename"].split("-")[2])
    fish_number = int(row["filename"].split("-")[3].split("(")[0].strip())
    row_match = oracle_df[(oracle_df.year==year) & (oracle_df.cruise_number==mission_number) & (oracle_df.fish_number==fish_number)]
    if row_match.shape[0] == 1:
        ages_writer.writerow([row["uuid"], row["filename"], int(row_match.iloc[0].age), int(row_match.iloc[0].annuli), int(row_match.iloc[0].edge_type), row_match.iloc[0].length, row_match.iloc[0].weight])
    elif row_match.shape[0] == 0:
        ages_writer.writerow([row["uuid"], row["filename"]])
    else:
        print(row)


def copy_annotated():
    SHARE_PATH = "/home/stoyelq/shares/otoliths_and_scales/Yellowtail-flounder"
    LINES_PATH = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/lines.csv"
    ANNOTATIONS_PATH = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/annotated/"
    lines_df = pd.read_csv(LINES_PATH)
    for root, dirs, files in os.walk(SHARE_PATH):
        for dir in dirs:
            for name in os.listdir(os.path.join(root, dir)):
                if name in lines_df["image_filename"].values:
                    src = os.path.join(root, dir, name)
                    dst = os.path.join(ANNOTATIONS_PATH, lines_df[lines_df.image_filename == name].iloc[0].annotation_uuid)
                    shutil.copy(src, f"{dst}.jpg")


def get_centers_from_line(uuid, num_images):
    LINES_PATH = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/lines.csv"
    lines_df = pd.read_csv(LINES_PATH)
    line_row = lines_df[lines_df.annotation_uuid == uuid].iloc[0]
    linestring = shapely.wkt.loads(line_row.annotation_preferred_linestring)
    coord_list = []
    for coord in linestring.coords:
        coord_list.append(coord)
    # x0, y0, x1, y1
    center_x, center_y, edge_x, edge_y= coord_list[0][0], coord_list[0][1], coord_list[1][0], coord_list[1][1]

    image_centers = []
    for i in range(num_images):
        # Calculate the interpolation factor (t)
        t = i / (num_images - 1)

        # Linearly interpolate x and y coordinates
        x = center_x + t * (edge_x - center_x)
        y = center_y + t * (edge_y - center_y)
        image_centers.append((int(x), int(y)))
    return image_centers


def get_dot_coord_list(uuid):
    DOTS_PATH = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/dots.csv"
    dots_df = pd.read_csv(DOTS_PATH)
    annotation_df = dots_df[dots_df.annotation_uuid == uuid][["x", "y"]].astype(int)
    dot_list = list(annotation_df.itertuples(index=False))
    return dot_list

def get_line_coord_list(uuid):
    LINES_PATH = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/lines.csv"
    lines_df = pd.read_csv(LINES_PATH)
    line_row = lines_df[lines_df.annotation_uuid == uuid].iloc[0]
    linestring = shapely.wkt.loads(line_row.annotation_preferred_linestring)
    coord_list = []
    for coord in linestring.coords:
        coord_list.append(coord)
    # x0, y0, x1, y1
    center_x, center_y, edge_x, edge_y = coord_list[0][0], coord_list[0][1], coord_list[1][0], coord_list[1][1]
    return center_x, center_y, edge_x, edge_y


def join_oracle_dump_to_metadata():
    metadata = pd.read_csv(RAW_METADATA)
    oracle_dump = pd.read_csv("/home/stoyelq/my_hot_storage/dfobot/yellowtail/yellowtail_oracle_dump.csv")

    with open("/home/stoyelq/my_hot_storage/dfobot/yellowtail/ages.csv",'w') as ages_sheet:
        ages_writer = csv.writer(ages_sheet)
        ages_writer.writerow(["uuid", "filename", "age", "annuli", "edge_type", "length", "weight"])

        metadata.apply(row_ager_and_writer, ages_writer=ages_writer, oracle_df=oracle_dump, axis=1)
    return


def wipe_ageless():
    target_dir = "/home/stoyelq/my_hot_storage/dfobot_working/ages/val/"
    img_list = os.listdir(target_dir)

    metadata_df = pd.read_csv("/home/stoyelq/my_hot_storage/dfobot_working/ages/ages.csv")

    for img_name in img_list:
        uuid = img_name.split(".")[0]
        metadata_row = metadata_df[(metadata_df["uuid"] == uuid)].iloc[0]
        if pd.isna(metadata_row.age):
            os.remove(os.path.join(target_dir, img_name))
    return

def create_annotation_images():
    in_dir = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/annotated/"
    out_dir = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/dot_images"
    num_images = 40
    image_size = 150  # half
    img_list = os.listdir(in_dir)

    annotation_sheet_path = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/annotations.csv"
    with open(annotation_sheet_path,'w') as annotations_sheet:
        sheet_writer = csv.writer(annotations_sheet)
        sheet_writer.writerow(["uuid", "is_dot"])

        for image_name in img_list:
            img_path = in_dir + image_name
            pil_img = Image.open(img_path).convert("RGB")

            img_uuid = image_name.split(".")[0]
            dot_list = np.array(get_dot_coord_list(img_uuid))
            centers_list = np.array(get_centers_from_line(img_uuid, num_images))

            # get the closest dot on the ref line to every annotation dot
            dot_centers = []
            for dot in dot_list:
                dists = np.linalg.norm(centers_list-dot, axis=1)
                min_index = np.argmin(dists)
                dot_centers.append(centers_list[min_index])

            for center in centers_list:
                new_uuid = str(uuid.uuid4())
                cropped = pil_img.crop((center[0] - image_size, center[1] - image_size, center[0] + image_size, center[1] + image_size))
                dst = os.path.join(out_dir, f"{new_uuid}.jpg")
                cropped.save(dst)
                sheet_writer.writerow([new_uuid, (center == dot_centers).all(axis=1).any()])
    return


def create_ref_line_images():
    in_dir = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/annotated/"
    out_dir = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/ref_line"
    img_list = os.listdir(in_dir)

    annotation_sheet_path = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/ref_line.csv"
    with open(annotation_sheet_path,'w') as annotations_sheet:
        sheet_writer = csv.writer(annotations_sheet)
        sheet_writer.writerow(["uuid", "center_x", "center_y", "edge_x", "edge_y"])

        for image_name in img_list:
            img_path = in_dir + image_name
            out_path = os.path.join(out_dir,image_name)
            pil_img = Image.open(img_path).convert("RGB")

            img_uuid = image_name.split(".")[0]
            line_coords = np.array(get_line_coord_list(img_uuid))

            new_coords = crop_and_transform(img_path, line_coords, out_path)
            sheet_writer.writerow([img_uuid, new_coords[0], new_coords[1], new_coords[2], new_coords[3]])
    return


def get_dfo_dots_ages(raw_df, dfo_dots_df, img_uuid):
    img_uuid = img_uuid.split(".")[0]
    img_name = raw_df.loc[raw_df["uuid"] == img_uuid].iloc[0]["filename"]

    year = int(img_name.split("-")[1])
    mission_number = int(img_name.split("-")[2])
    fish_number = int(img_name.split("-")[3].split("(")[0].strip())

    matching_rows = dfo_dots_df.loc[(dfo_dots_df["collection_year"] == year) & (dfo_dots_df["fish_number"] == fish_number)]
    if len(matching_rows) == 1:
        # uuid,filename,age,annuli,edge_type,length,weight
        img_uuid = uuid.uuid4()
        return [img_uuid, img_name, matching_rows["age_manual"].iloc[0], matching_rows["annulus_count"].iloc[0], matching_rows["edge_type"].iloc[0], matching_rows["length_mm"].iloc[0], matching_rows["weight_g"].iloc[0]]
    elif len(matching_rows) == 0:
        return False
    else:
        print(f"multiple matches for {img_name}")
        return False

def get_oracle_ages(raw_df, oracle_df,img_uuid):
    img_uuid = img_uuid.split(".")[0]
    img_name = raw_df.loc[raw_df["uuid"] == img_uuid].iloc[0]["filename"]

    year = int(img_name.split("-")[1])
    mission_number = int(img_name.split("-")[2])
    fish_number = int(img_name.split("-")[3].split("(")[0].strip())

    matching_rows = oracle_df.loc[(oracle_df["year"] == year) & (oracle_df["cruise_number"] == mission_number) & (oracle_df["fish_number"] == fish_number)]
    if len(matching_rows) == 1:
        # uuid,filename,age,annuli,edge_type,length,weight
        img_uuid = uuid.uuid4()
        return [img_uuid, img_name, matching_rows["age"].iloc[0], matching_rows["annuli"].iloc[0], matching_rows["edge_type"].iloc[0], matching_rows["length"].iloc[0], matching_rows["weight"].iloc[0]]
    elif len(matching_rows) == 0:
        return False
    else:
        print(f"multiple matches for {img_name}")
        return False

def make_combined_ages():
    img_list = os.listdir(RAW_DIR)
    img_name_df = pd.read_csv(RAW_METADATA)
    dfo_dots_df = pd.read_csv(DFODOTS_METADATA)
    oracle_df = pd.read_csv(ORACLE_METADATA)

    in_dir = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/raw/"
    out_dir = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/ages_combined/"

    dfo_dots_count = 0
    oracle_count = 0
    row_index = 0

    with open("/home/stoyelq/my_hot_storage/dfobot/yellowtail/combined_ages.csv",'w') as ages_sheet:
        ages_writer = csv.writer(ages_sheet)
        ages_writer.writerow(["uuid", "filename", "age", "annuli", "edge_type", "length", "weight"])
    
        for img_name in img_list:
            age_data = get_dfo_dots_ages(img_name_df, dfo_dots_df, img_name)
            if age_data:
                dfo_dots_count += 1
            else:
                age_data = get_oracle_ages(img_name_df, oracle_df, img_name)
                oracle_count += 1

            if age_data:
                ages_writer.writerow(age_data)
                src = os.path.join(in_dir, img_name)
                dst = os.path.join(out_dir, f"{age_data[0]}.jpg")
                shutil.copy(src, dst)
        print(dfo_dots_count)
        print(oracle_count)



#
# DATA_DIR = "/home/stoyelq/Documents/dfobot_data/plaice/"
# crop_and_isolate(herring=False)
# DATA_DIR = "/home/stoyelq/Documents/dfobot_data/herring/enhanced/"
# crop_and_isolate()
# #
# IN_DIR = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/ages_combined"


# import_crab()
# make_combined_ages()
OUT_DIR = "/home/stoyelq/my_hot_storage/dfobot_working/crab/classes/"
IN_DIR = "/home/stoyelq/my_hot_storage/dfobot/crab/classed"
train_val_splitter(IN_DIR, OUT_DIR, split=0.8)

# wipe_ageless()
# copy_annotated()

# create_annotation_images()
# create_ref_line_images()
#
