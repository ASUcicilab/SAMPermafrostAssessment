#!/usr/bin/env python

"""
This script preprocess data into detectron2's dataset format
the dataset is in list[dict], each dict contains information about one image.
The dict have the following fields:

* file_name: the full path to the image file.
* height, width: integer. The shape of the image.
* image_id (str or int): a unique id that identifies this image. Required by many evaluators to identify the images, but a dataset may use it for different purposes.
* annotations (list[dict]): Each dict corresponds to annotations of one instance in this image, and contains the following keys:
    bbox (list[float], required): list of 4 numbers representing the bounding box of the instance.
    bbox_mode (int, required): the format of bbox. It must be a member of structures.BoxMode. Currently supports: BoxMode.XYXY_ABS, BoxMode.XYWH_ABS.
    category_id (int, required): an integer in the range [0, num_categories-1] representing the category label. The value num_categories is reserved to represent the “background” category, if applicable.
    segmentation (list[list[float]] or dict): the segmentation mask of the instance.
    list[list[float]] represents a list of polygons, one for each connected component of the object. Each list[float] is one simple polygon in the format of [x1, y1, ..., xn, yn] (n≥3). The Xs and Ys are absolute coordinates in unit of pixels.

If annotations is an empty list, it means the image is labeled to have no objects. Such images will by default be removed from training, but can be included using DATALOADER.FILTER_EMPTY_ANNOTATIONS.

"""

import argparse
import json
import os

import cv2
import numpy as np
from detectron2.structures import BoxMode


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def preprocess_iwp_helper(data_dir, split):
    output = []

    # load the json file
    with open(os.path.join("data", data_dir, split, "via_region_data.json"), "r") as f:
        data = json.load(f)

    # iterate through each image
    for image_id, value in enumerate(data.values()):
        image_path = os.path.join("data", data_dir, split, value["filename"])
        height, width = cv2.imread(image_path).shape[:2]
        annotations = value["regions"]

        # create the annotation list for all annotations in the image
        annotation_list = []
        # iterate through each annotation
        for annotation in annotations:
            region = annotation["region_attributes"]

            # get the category id
            category_id = None
            if region["object_name"] == "highcenter":
                category_id = 0
            elif region["object_name"] == "lowcenter":
                category_id = 0
            else:
                continue
            assert category_id is not None, "Invalid category id: {}".format(
                category_id
            )

            # get the polygon
            shape = annotation["shape_attributes"]
            px = shape["all_points_x"]
            py = shape["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            # create the annotation dict
            annotation_dict = {
                # make bbox JSON serializable and add 1 to the coordinates
                "bbox": [
                    int(np.min(px)) - 1,
                    int(np.min(py)) - 1,
                    int(np.max(px)) + 1,
                    int(np.max(py)) + 1,
                ],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": category_id,
                "segmentation": [poly],
            }

            # append the annotation dict to the annotation list
            annotation_list.append(annotation_dict)

        # create the image dict
        image_dict = {
            "file_name": image_path,
            "height": height,
            "width": width,
            "image_id": image_id,
            "annotations": annotation_list,
        }

        # append the image dict to the output list
        output.append(image_dict)

    # return the output list
    return output


def preprocess_iwp(dataset_name):
    # combine train and val into train
    train = preprocess_iwp_helper(dataset_name, "train")
    val = preprocess_iwp_helper(dataset_name, "val")
    train.extend(val)
    # save the train and val lists to json files in the dataset folder
    with open(os.path.join("data", dataset_name, "iwp_train.json"), "w") as f:
        json.dump(train, f)

    # create the test list and save it to a json file in the dataset folder
    test = preprocess_iwp_helper(dataset_name, "test")
    with open(os.path.join("data", dataset_name, "iwp_test.json"), "w") as f:
        json.dump(test, f)


def generate_rts_images():
    # load foler names with 'MAXAR' in the name
    folder_names = [
        folder_name for folder_name in os.listdir("data/rts") if "MAXAR" in folder_name
    ]

    # iterate through each folder
    for folder_name in folder_names:
        # load .npy file names in the folder without 'mask' in the name
        file_names = [
            file_name
            for file_name in os.listdir(os.path.join("data/rts", folder_name))
            if ".npy" in file_name and "mask" not in file_name
        ]
        # iterate through each file
        for file_name in file_names:
            # load the file
            image = np.load(os.path.join("data/rts", folder_name, file_name))
            # continue if the image has 4 dimensions
            if len(image.shape) >= 4:
                continue
            # take the first three channels
            image = image[:, :, :3]
            # scale to 0-255 and convert to uint8
            image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
            image = image.astype(np.uint8)
            # transfer the image from RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # save the image to a .png file with the folder name and file name
            # the destination folder is test folder if the folder name contains 'VALTEST' otherwise it is train folder
            if "VALTEST" in folder_name:
                cv2.imwrite(
                    os.path.join(
                        "data/rts", "test", folder_name + "_" + file_name[:-4] + ".png"
                    ),
                    image,
                )
            else:
                cv2.imwrite(
                    os.path.join(
                        "data/rts", "train", folder_name + "_" + file_name[:-4] + ".png"
                    ),
                    image,
                )


def preprocess_rts_helper(data_dir, split):
    # create the output list
    output = []
    # load image names in the split folder
    image_names = os.listdir(os.path.join("data", data_dir, split))
    # iterate through each image
    for image_id, image_name in enumerate(image_names):
        # get the image path
        image_path = os.path.join("data", data_dir, split, image_name)
        # get the image height and width
        height, width = cv2.imread(image_path).shape[:2]
        # get the annotations
        annotations = []
        # read the annotation npy file
        # for an image name 'MAXAR_YG_T1_030.png', the annotation file is in the folder 'MAXAR_YG_T1' and the file name is '030.mask.npy'
        annotation_path = os.path.join(
            "data", data_dir, image_name[:-8], image_name[-7:-4] + ".mask.npy"
        )
        # load the annotation file
        annotation = np.load(annotation_path)
        # iterate through the first channel of the annotation and save the annotation to the annotations list
        for i in range(annotation.shape[0]):
            annotation_list = {}
            # get the annotation mask
            mask = annotation[i, :, :]
            # get the bounding box
            bbox = cv2.boundingRect(mask)
            # save the bounding box to the annotation list as xyxy
            annotation_list["bbox"] = [
                bbox[0],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3],
            ]
            # save the bounding box mode to the annotation list
            annotation_list["bbox_mode"] = 0
            # save the category id to the annotation list
            annotation_list["category_id"] = 0
            # transfer the mask to polygon
            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # get the contour with the largest area
            contour = max(contours, key=lambda x: PolyArea(x[:, 0, 0], x[:, 0, 1]))
            # save the contour to the annotation list
            annotation_list["segmentation"] = [contour[:, 0, :].flatten().tolist()]
            if len(annotation_list["segmentation"][0]) < 6:
                print(
                    "Invalid annotation: {}".format(annotation_list["segmentation"][0])
                )
                continue
            # append the annotation list to the annotations
            annotations.append(annotation_list)

        # create the image dict
        image_dict = {
            "file_name": image_path,
            "height": height,
            "width": width,
            "image_id": image_id,
            "annotations": annotations,
        }

        # append the image dict to the output list
        output.append(image_dict)

    # return the output list
    return output


def preprocess_rts(dataset_name):
    generate_rts_images()

    # preprocess the train split
    train = preprocess_rts_helper(dataset_name, "train")

    # save the train list to json files in the dataset folder
    with open(os.path.join("data", dataset_name, "rts_train.json"), "w") as f:
        json.dump(train, f)

    # create the test list
    test = preprocess_rts_helper(dataset_name, "test")

    # save the test list to a json file in the dataset folder
    with open(os.path.join("data", dataset_name, "rts_test.json"), "w") as f:
        json.dump(test, f)


def preprocess_data(dataset_name):
    if dataset_name == "iwp":
        preprocess_iwp(dataset_name)
    if dataset_name == "rts":
        preprocess_rts(dataset_name)
    else:
        raise ValueError("Invalid dataset name: {}".format(dataset_name))

    print("Preprocessing {} complete.".format(dataset_name))


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Preprocess data for a given dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="folder name of the dataset being processed",
    )
    args = parser.parse_args()

    preprocess_data(args.dataset)
