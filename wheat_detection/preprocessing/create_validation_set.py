import os
import glob
import yaml
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


def get_image_id(path):
    _, name = os.path.split(path)
    return name[:-4]


if __name__ == "__main__":
    # read config
    with open("config.yaml") as config_file:
        config = yaml.full_load(config_file)

    # read bounding boxes data
    boxes_df = pd.read_csv(os.path.join(config["data_path"], "bounding_boxes.csv"))

    # split images into train/validation parts
    img_list = glob.glob(os.path.join(config["data_path"], "images", '*.jpg'))
    train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=42)

    # copy images and bounding boxes file into appropriate folders
    for img_list, folder_name in zip([train_img_list, val_img_list], ["train", "val"]):
        os.makedirs(os.path.join(config["output_path"], folder_name, "images"))

        df = None
        for img in img_list:
            img_id = get_image_id(img)
            df_cur = boxes_df[boxes_df["image_id"] == img_id]
            df = df_cur if df is None else pd.concat((df, df_cur), axis=0)

        df.to_csv(os.path.join(config["output_path"], folder_name, "bounding_boxes.csv"), index=False)

        for img_path in img_list:
            _, img_name = os.path.split(img_path)
            shutil.copy(img_path, os.path.join(config["output_path"], folder_name, "images", img_name))
