import os
import cv2
import torch
import numpy as np
import pandas as pd

from wheat_detection.model_training.augmentations import get_transforms


base_transform_config = {"size": 512,
                         "min_visibility": 0.2,
                         "names": ["resize"]}
base_transform = get_transforms(base_transform_config)


class WheatDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, drop_empty=False):
        self.imgs_path = os.path.join(path, "images")
        self.transform = transform
        self.drop_empty = drop_empty

        self.imgs = os.listdir(self.imgs_path)
        self.boxes_df = pd.read_csv(os.path.join(path, "bounding_boxes.csv"))

        if drop_empty:
            self.imgs = [img for img in self.imgs if self.boxes_df[self.boxes_df["image_id"] == img.replace(".jpg", "")].shape[0] > 0]

    def __getitem__(self, idx):
        # read image
        img_name = self.imgs[idx]
        img_path = os.path.join(self.imgs_path, img_name)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # get bounding box coordinates
        boxes_str = list(self.boxes_df[self.boxes_df["image_id"] == img_name.replace(".jpg", "")]["bbox"])
        bboxes = [list(map(float, box[1:-1].split(","))) for box in boxes_str]
        bboxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in bboxes]
        bboxes = np.array(bboxes, dtype=np.float32)

        # there is only one class
        labels = np.ones((bboxes.shape[0],), dtype=np.int64)

        # transform image and bboxes
        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=bboxes, class_labels=labels)
            if self.drop_empty and len(transformed["bboxes"]) == 0:
                transformed = base_transform(image=img, bboxes=bboxes, class_labels=labels)
            img, bboxes, labels = transformed["image"], transformed["bboxes"], transformed["class_labels"]

        num_objects = len(bboxes)
        # convert numpy arrays to PyTorch tensors
        if num_objects != 0:
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        else:
            bboxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            area = torch.empty((0,), dtype=torch.float32)

        # define variables for PyTorch Faster R-CNN model compatibility
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((num_objects,), dtype=torch.int64)

        target = {"boxes": bboxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        img = np.true_divide(img, 255, dtype=np.float32)
        return torch.from_numpy(img).permute(2, 0, 1), target

    def __len__(self):
        # return 32
        return len(self.imgs)
