#!/usr/bin/env python
"""
this script evaluates sam model on the test set using ground truth bbox or points as the prompt
the current metric numbers are:
iwp:
bbox: 0.844
points: 0.233
rts:
bbox: 0.804
points: 0.085
"""

import json
import argparse
import cv2
import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from detectron2.structures import Instances, Boxes


def get_data_dicts(dataset, split):
    """
    Load dataset JSON file from the dataset_name
    """
    dataset_dicts = json.load(
        open("data/{}/{}_{}.json".format(dataset, dataset, split))
    )
    return dataset_dicts


def sam_no_finetune(args):
    # register dataset
    DatasetCatalog.register(
        "{}_test".format(args.dataset), lambda: get_data_dicts(args.dataset, "test")
    )
    MetadataCatalog.get("{}_test".format(args.dataset)).set(
        thing_classes=[args.dataset]
    )
    dataset = DatasetCatalog.get("{}_test".format(args.dataset))

    # load sam model
    sam = sam_model_registry["vit_h"](
        checkpoint="pretrained_weights/sam_vit_h_4b8939.pth"
    )
    sam.to("cuda:0")
    sam.eval()

    # build evaluator
    evaluator = COCOEvaluator(
        "{}_test".format(args.dataset),
        output_dir="experiments/sam_no_finetune/results/{}/{}/".format(
            args.dataset, args.prompt
        ),
        max_dets_per_image=args.max_dets_per_image,
    )
    evaluator.reset()

    # transformation for sam input
    transform = ResizeLongestSide(sam.image_encoder.img_size)

    # iterate through the dataset
    for idx, data in enumerate(dataset):
        # print progress every 10 images
        if idx % 10 == 0:
            print("Processing image {}/{}".format(idx, len(dataset)))

        # create input for SAM
        sam_input = [{}]
        # image input
        img = cv2.imread(data["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sam_input[0]["original_size"] = (img.shape[0], img.shape[1])
        img = transform.apply_image(img)
        # change the shape to (3, H, W)
        sam_input[0]["image"] = (
            torch.as_tensor(img.astype("float32")).permute(2, 0, 1).to("cuda:0")
        )

        # points input
        point_coords = []
        boxes = []
        for anno in data["annotations"]:
            # calculate the center of the bbox
            boxes.append(anno["bbox"])
            x1, y1, x2, y2 = anno["bbox"]
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            point_coords.append([x, y])

        if args.prompt == "bbox":
            boxes = torch.as_tensor(boxes).to("cuda:0")
            transformed_boxes = transform.apply_boxes_torch(
                boxes, original_size=sam_input[0]["original_size"]
            )
            sam_input[0]["boxes"] = transformed_boxes
        elif args.prompt == "point":
            # transfer to numpy with shape (N, 2)
            point_coords = np.array(point_coords)
            point_coords = transform.apply_coords(
                point_coords, original_size=sam_input[0]["original_size"]
            )
            point_coords = torch.as_tensor(point_coords).unsqueeze(1).to("cuda:0")
            sam_input[0]["point_coords"] = point_coords
            # creata point labels for SAM with shape (N,1)
            sam_input[0]["point_labels"] = torch.ones(len(point_coords), 1).to("cuda:0")
        else:
            # raise error if the prompt is not supported
            raise ValueError("Prompt {} is not supported".format(args.prompt))

        # run SAM
        with torch.no_grad():
            sam_output = sam(sam_input, multimask_output=False, required_grad=False)

        # create input and output for evaluator
        eval_input = [{}]
        eval_input[0]["file_name"] = data["file_name"]
        eval_input[0]["image_id"] = data["image_id"]
        eval_input[0]["height"] = data["height"]
        eval_input[0]["width"] = data["width"]

        eval_output = [{}]
        eval_output[0]["instances"] = Instances((data["height"], data["width"]))
        eval_output[0]["instances"].pred_masks = sam_output[0]["masks"].squeeze(1)
        eval_output[0]["instances"].pred_classes = torch.zeros(
            len(sam_output[0]["masks"]), dtype=torch.int64
        ).to("cuda:0")
        eval_output[0]["instances"].scores = sam_output[0]["iou_predictions"].squeeze(1)
        eval_output[0]["instances"].pred_boxes = Boxes(boxes)

        evaluator.process(eval_input, eval_output)

    results = evaluator.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SAM on the test set of the specified dataset with ground truth bbox/point as prompt."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="name of the dataset, default is dataset",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="bbox",
        help="prompt for SAM, bbox or point, default is bbox",
    )
    args = parser.parse_args()

    # costomize the max_dets_per_image
    if args.dataset == "iwp":
        args.max_dets_per_image = 500
    elif args.dataset == "rts":
        args.max_dets_per_image = 100
    else:
        # raise error if the dataset is not supported
        raise ValueError("Dataset {} is not supported.".format(args.dataset))

    sam_no_finetune(args)
