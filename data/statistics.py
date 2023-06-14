#!/usr/bin/env python
"""
This script is to calculate the statistics of the dataset.
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def generate_statistics(args):
    # load the train data
    data_train = json.load(
        open("data/{}/{}_train.json".format(args.dataset, args.dataset), "r")
    )
    # load the test data
    data_test = json.load(
        open("data/{}/{}_test.json".format(args.dataset, args.dataset), "r")
    )

    # get the number of images
    num_images_train = len(data_train)
    num_images_test = len(data_test)
    print("Number of images in the train set: {}".format(num_images_train))
    print("Number of images in the test set: {}".format(num_images_test))

    # get the number of annotations
    num_annotations_train = 0
    num_annotations_test = 0
    for sample in data_train:
        num_annotations_train += len(sample["annotations"])
    for sample in data_test:
        num_annotations_test += len(sample["annotations"])
    print("Number of annotations in the train set: {}".format(num_annotations_train))
    print("Number of annotations in the test set: {}".format(num_annotations_test))

    # plot the box area distribution of the train and test set on a 2x1 subplot
    box_areas_train = []
    box_areas_test = []
    for sample in data_train:
        for annotation in sample["annotations"]:
            box_areas_train.append(
                (annotation["bbox"][3] - annotation["bbox"][1])
                * (annotation["bbox"][2] - annotation["bbox"][0])
            )
    for sample in data_test:
        for annotation in sample["annotations"]:
            box_areas_test.append(
                (annotation["bbox"][3] - annotation["bbox"][1])
                * (annotation["bbox"][2] - annotation["bbox"][0])
            )
    plt.subplot(2, 1, 1)
    plt.hist(box_areas_train, bins=100)
    plt.title("Box Area Distribution of the Train Set")
    plt.xlabel("Box Area")
    plt.ylabel("Number of Boxes")
    plt.subplot(2, 1, 2)
    plt.hist(box_areas_test, bins=100)
    plt.title("Box Area Distribution of the Test Set")
    plt.xlabel("Box Area")
    plt.ylabel("Number of Boxes")
    plt.tight_layout()
    plt.savefig("data/{}/box_area_distribution.png".format(args.dataset))
    plt.close()

    # plot the mask area distribution of the train and test set on a 2x1 subplot
    # the mask format is in polygon format
    mask_areas_train = []
    mask_areas_test = []
    for sample in data_train:
        for annotation in sample["annotations"]:
            # calculate the area of the mask from the polygon format using the shoelace formula
            # the polygon format is x1, y1, x2, y2, ..., xn, yn
            mask = annotation["segmentation"][0]
            mask = np.array(mask).reshape(-1, 2) 
            mask_area = 0.0
            for i in range(mask.shape[0] - 1):
                mask_area += mask[i][0] * mask[i + 1][1]
                mask_area -= mask[i + 1][0] * mask[i][1]
            mask_area = abs(mask_area) / 2.0
            mask_areas_train.append(mask_area)
    for sample in data_test:
        for annotation in sample["annotations"]:
            # calculate the area of the mask from the polygon format using the shoelace formula
            # the polygon format is x1, y1, x2, y2, ..., xn, yn
            mask = annotation["segmentation"][0]
            mask = np.array(mask).reshape(-1, 2)
            mask_area = 0.0
            for i in range(mask.shape[0] - 1):
                mask_area += mask[i][0] * mask[i + 1][1]
                mask_area -= mask[i + 1][0] * mask[i][1]
            mask_area = abs(mask_area) / 2.0
            mask_areas_test.append(mask_area)
    # plot the mask area distribution of the train and test set on a 2x1 subplot and make the x-axis the same scale
    plt.subplot(2, 1, 1)
    plt.hist(mask_areas_train, bins=100)
    plt.title("Mask Area Distribution of the Train Set")
    plt.xlabel("Mask Area")
    plt.ylabel("Number of Masks")
    # make the x-axis the same scale
    plt.xlim(0, 30000)
    plt.subplot(2, 1, 2)
    plt.hist(mask_areas_test, bins=100)
    plt.title("Mask Area Distribution of the Test Set")
    plt.xlabel("Mask Area")
    plt.ylabel("Number of Masks")
    # make the x-axis the same scale
    plt.xlim(0, 30000)
    plt.tight_layout()
    plt.savefig("data/{}/mask_area_distribution.png".format(args.dataset))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the statistics of the dataset."
    )
    parser.add_argument("--dataset", type=str, help="name of the dataset")
    args = parser.parse_args()

    generate_statistics(args)
