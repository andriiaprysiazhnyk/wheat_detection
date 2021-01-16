import cv2
import albumentations as albu


def get_transforms(config):
    size = config["size"]
    min_visibility = config["min_visibility"]

    mapping = {"crop_or_resize": albu.OneOf([albu.RandomCrop(size, size), albu.Resize(height=size, width=size)], p=1),
               "resize": albu.Resize(height=size, width=size, interpolation=cv2.INTER_AREA),
               "flip": albu.HorizontalFlip(),
               "change_brightness": albu.RandomBrightnessContrast(p=0.2)}

    return albu.Compose([mapping[transform_name] for transform_name in config["names"]],
                        bbox_params=albu.BboxParams(format="pascal_voc",
                                                    min_visibility=min_visibility,
                                                    label_fields=["class_labels"]))
