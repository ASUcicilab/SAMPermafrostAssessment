#!/usr/bin/env python

"""
This script generates samples from the dataset with masks and bboxes.

with the following command:
python generate_samples.py --dataset dataset --split train --num_samples 2
The generated samples are saved in data/samples/dataset_name folder.
"""

import argparse
import json
import os

import cv2
import numpy as np


def generate_samples(dataset_name, split, num_samples):
    """
    Generate samples from the dataset with masks and bboxes.

    Args:
        dataset_name (str): name of the dataset
        split (str): train, val, or test
        num_samples (int): number of samples to generate

    Returns:
        None
    """

    # load the data
    data = json.load(
        open("data/{}/{}_{}.json".format(dataset_name, dataset_name, split), "r")
    )

    # randomly select num_samples from the data
    if num_samples == 'all':
        num_samples = len(data)
    else:
        num_samples = int(num_samples)

    samples = np.random.choice(data, num_samples, replace=False)

    # iterate through each sample
    for sample in samples:
        # read the image
        image_path = sample["file_name"]
        image = cv2.imread(image_path)

        # iterate through each annotation
        annotations = sample["annotations"]
        for annotation in annotations:
            # get the bbox and mask
            bbox = annotation["bbox"]
            mask = annotation["segmentation"]

            # convert the mask to a list of numpy array
            mask = np.array(mask).reshape(-1, 2).astype(np.int32)
            mask = [mask]

            # draw the bbox and mask
            image = cv2.rectangle(
                image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1
            )
            image = cv2.drawContours(image, mask, -1, (0, 0, 255), 1)

        # save the image to a samples/dataset_name folder
        if not os.path.exists("data/samples/{}".format(dataset_name)):
            os.makedirs("data/samples/{}".format(dataset_name))
        cv2.imwrite(
            "data/samples/{}/{}".format(dataset_name, os.path.basename(image_path)),
            image,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from the dataset.")
    parser.add_argument(
        "--dataset", type=str, default="dataset", help="name of the dataset"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="train, val, or test"
    )
    parser.add_argument(
        "--num_samples", type=str, default='all', help="number of samples to generate"
    )
    args = parser.parse_args()

    generate_samples(args.dataset, args.split, args.num_samples)
