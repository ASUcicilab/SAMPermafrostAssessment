#!/usr/bin/env python
"""
this script is used to generate predictions for all models in experiments
"""
import argparse
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from detectron2.config import get_cfg


def maskrcnn(data):
    args = argparse.Namespace()
    # hyperparameters for prediction
    # dataset
    args.dataset = "iwp"
    args.things_classes = ["IWP"]  # classes to be detected
    args.config_file = "config_files/mask_rcnn_R_50_FPN_3x.yaml"
    args.num_classes = 1
    args.seed = 1129
    args.detections_per_image = 500
    args.output_dir = "experiments/sam_maskrcnn_no_finetune/results/{}".format(
        args.dataset
    )
    args.model_weight = "experiments/maskrcnn/results/{}/model_{}.pth".format(
        args.dataset, "0001199"
    )
    # model with default config
    args.min_size_train = (640, 672, 704, 736, 768, 800)
    args.max_size_train = 1333
    args.min_size_test = 800
    args.max_size_test = 1333
    args.anchor_sizes = [[32, 64, 128, 256, 512]]
    args.device = "cuda:0"

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    # model config
    cfg.MODEL.WEIGHTS = args.model_weight
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.INPUT.MIN_SIZE_TRAIN = args.min_size_train
    cfg.INPUT.MAX_SIZE_TRAIN = args.max_size_train
    cfg.INPUT.MIN_SIZE_TEST = args.min_size_test
    cfg.INPUT.MAX_SIZE_TEST = args.max_size_test
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = args.anchor_sizes
    cfg.SEED = args.seed
    cfg.TEST.DETECTIONS_PER_IMAGE = args.detections_per_image
    cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.DEVICE = args.device

    # create predictor
    from detectron2.engine import DefaultPredictor

    predictor = DefaultPredictor(cfg)

    # predict
    img = cv2.imread(data["file_name"])
    outputs = predictor(img)

    # iterate through all masks
    for i in range(outputs["instances"].pred_masks.shape[0]):
        mask = outputs["instances"].pred_masks[i].cpu().numpy()
        # transfer the mask to polygon
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        polygon = contours[0].reshape((-1, 2)).tolist()
        # draw the polygon on the image
        img = cv2.polylines(img, [np.array(polygon)], True, (255, 0, 0), 2)

    return img


def sam_no_finetune(data):
    # use ground truth boxes to run sam
    # load sam model
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry["vit_h"](
        checkpoint="pretrained_weights/sam_vit_h_4b8939.pth"
    )
    predictor = SamPredictor(sam)

    # read image
    img = cv2.imread(data["file_name"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # load image into sam
    predictor.set_image(img)

    # iterate through all annotations
    for anno in data["annotations"]:
        # load the box
        box = anno["bbox"]
        box = np.array(box).astype(np.int32)
        # run sam
        mask, score, _ = predictor.predict(box=box, multimask_output=False)
        mask = mask.squeeze(0)
        # transfer the mask to polygon
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        polygon = contours[0].reshape((-1, 2)).tolist()
        # draw the polygon on the image
        img = cv2.polylines(img, [np.array(polygon)], True, (255, 0, 0), 2)

    return img


def sam_no_finetune_with_points(data):
    # use ground truth boxes to run sam
    # load sam model
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry["vit_h"](
        checkpoint="pretrained_weights/sam_vit_h_4b8939.pth"
    )
    predictor = SamPredictor(sam)

    # read image
    img = cv2.imread(data["file_name"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # load image into sam
    predictor.set_image(img)

    # iterate through all annotations
    for anno in data["annotations"]:
        # load the box
        box = anno["bbox"]
        # find the center of the box and transfer to numpy array with shape (1, 2)
        point_coords = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        point_coords = point_coords.reshape((1, 2))
        point_labels = np.array([1])

        # run sam
        mask, score, _ = predictor.predict(
            point_coords=point_coords, point_labels=point_labels, multimask_output=False
        )
        mask = mask.squeeze(0)
        # transfer the mask to polygon
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        polygon = contours[0].reshape((-1, 2)).tolist()
        # draw the polygon on the image
        img = cv2.polylines(img, [np.array(polygon)], True, (255, 0, 0), 2)

    return img


def sam_with_finetune(data):
    # use ground truth boxes to run sam
    # load sam model
    from detectron2.checkpoint import DetectionCheckpointer
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry["vit_h"]()
    DetectionCheckpointer(sam).load(
        "experiments/sam_with_finetune/results/iwp/model_0000139.pth"
    )
    predictor = SamPredictor(sam)
    # read image
    img = cv2.imread(data["file_name"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # load image into sam
    predictor.set_image(img)

    # iterate through all annotations
    for anno in data["annotations"]:
        # load the box
        box = anno["bbox"]
        box = np.array(box).astype(np.int32)
        # run sam
        mask, score, _ = predictor.predict(box=box, multimask_output=False)
        mask = mask.squeeze(0)
        # transfer the mask to polygon
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        polygon = contours[0].reshape((-1, 2)).tolist()
        # draw the polygon on the image
        img = cv2.polylines(img, [np.array(polygon)], True, (255, 0, 0), 2)

    return img


def sam_maskrcnn_without_finetune(data):
    args = argparse.Namespace()
    # hyperparameters for prediction
    # dataset
    args.dataset = "iwp"
    args.things_classes = ["IWP"]  # classes to be detected
    args.config_file = "config_files/mask_rcnn_R_50_FPN_3x.yaml"
    args.num_classes = 1
    args.seed = 1129
    args.detections_per_image = 500
    args.output_dir = "experiments/sam_maskrcnn_no_finetune/results/{}".format(
        args.dataset
    )
    args.model_weight = "experiments/maskrcnn/results/{}/model_{}.pth".format(
        args.dataset, "0001199"
    )
    # model with default config
    args.min_size_train = (640, 672, 704, 736, 768, 800)
    args.max_size_train = 1333
    args.min_size_test = 800
    args.max_size_test = 1333
    args.anchor_sizes = [[32, 64, 128, 256, 512]]
    args.device = "cuda:0"

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    # model config
    cfg.MODEL.WEIGHTS = args.model_weight
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.INPUT.MIN_SIZE_TRAIN = args.min_size_train
    cfg.INPUT.MAX_SIZE_TRAIN = args.max_size_train
    cfg.INPUT.MIN_SIZE_TEST = args.min_size_test
    cfg.INPUT.MAX_SIZE_TEST = args.max_size_test
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = args.anchor_sizes
    cfg.SEED = args.seed
    cfg.TEST.DETECTIONS_PER_IMAGE = args.detections_per_image
    cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.DEVICE = args.device

    # create predictor
    from detectron2.engine import DefaultPredictor

    predictor = DefaultPredictor(cfg)

    # predict
    img = cv2.imread(data["file_name"])
    outputs = predictor(img)

    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry["vit_h"](
        checkpoint="pretrained_weights/sam_vit_h_4b8939.pth"
    )
    predictor = SamPredictor(sam)

    # read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # load image into sam
    predictor.set_image(img)

    boxes = outputs["instances"].pred_boxes.tensor

    # iterate through all annotations
    for i in range(boxes.shape[0]):
        box = boxes[i].cpu().numpy()
        box = box.astype(np.int32)
        # run sam
        mask, score, _ = predictor.predict(box=box, multimask_output=False)
        mask = mask.squeeze(0)
        # transfer the mask to polygon
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        polygon = contours[0].reshape((-1, 2)).tolist()
        # draw the polygon on the image
        img = cv2.polylines(img, [np.array(polygon)], True, (255, 0, 0), 2)

    return img


def sam_maskrcnn_with_finetune(data):
    args = argparse.Namespace()
    # hyperparameters for prediction
    # dataset
    args.dataset = "iwp"
    args.things_classes = ["IWP"]  # classes to be detected
    args.config_file = "config_files/mask_rcnn_R_50_FPN_3x.yaml"
    args.num_classes = 1
    args.seed = 1129
    args.detections_per_image = 500
    args.output_dir = "experiments/sam_maskrcnn_no_finetune/results/{}".format(
        args.dataset
    )
    args.model_weight = "experiments/maskrcnn/results/{}/model_{}.pth".format(
        args.dataset, "0001199"
    )
    # model with default config
    args.min_size_train = (640, 672, 704, 736, 768, 800)
    args.max_size_train = 1333
    args.min_size_test = 800
    args.max_size_test = 1333
    args.anchor_sizes = [[32, 64, 128, 256, 512]]
    args.device = "cuda:0"

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    # model config
    cfg.MODEL.WEIGHTS = args.model_weight
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.INPUT.MIN_SIZE_TRAIN = args.min_size_train
    cfg.INPUT.MAX_SIZE_TRAIN = args.max_size_train
    cfg.INPUT.MIN_SIZE_TEST = args.min_size_test
    cfg.INPUT.MAX_SIZE_TEST = args.max_size_test
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = args.anchor_sizes
    cfg.SEED = args.seed
    cfg.TEST.DETECTIONS_PER_IMAGE = args.detections_per_image
    cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.DEVICE = args.device

    # create predictor
    from detectron2.engine import DefaultPredictor

    predictor = DefaultPredictor(cfg)

    # predict
    img = cv2.imread(data["file_name"])
    outputs = predictor(img)

    from detectron2.checkpoint import DetectionCheckpointer
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry["vit_h"]()
    DetectionCheckpointer(sam).load(
        "experiments/sam_with_finetune/results/iwp/model_0000139.pth"
    )
    predictor = SamPredictor(sam)

    # read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # load image into sam
    predictor.set_image(img)

    boxes = outputs["instances"].pred_boxes.tensor

    # iterate through all annotations
    for i in range(boxes.shape[0]):
        box = boxes[i].cpu().numpy()
        box = box.astype(np.int32)
        # run sam
        mask, score, _ = predictor.predict(box=box, multimask_output=False)
        mask = mask.squeeze(0)
        # transfer the mask to polygon
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        polygon = contours[0].reshape((-1, 2)).tolist()
        # draw the polygon on the image
        img = cv2.polylines(img, [np.array(polygon)], True, (255, 0, 0), 2)

    return img


def generate_predictions(args):
    # load test json file
    test_json = json.load(
        open("data/{}/{}_test.json".format(args.dataset, args.dataset))
    )
    # iterate through all data in the test json file
    for idx, data in enumerate(test_json):
        # print progress
        print("processing {}/{}".format(idx, len(test_json)))

        # plot a 2x3 figure
        # fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        # plot a 1x2 figure
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # plot first figure
        # load the image
        img = cv2.imread(data["file_name"])
        # load all the polygons and plot the polygons on the image
        for anno in data["annotations"]:
            polygon = anno["segmentation"]
            polygon = np.array(polygon).reshape((-1, 2)).astype(np.int32)
            img = cv2.polylines(img, [polygon], True, (255, 0, 0), 2)

        # plot the image
        axs[0].imshow(img)
        axs[0].set_title("ground truth")
        axs[0].axis("off")

        # plot second figure with sam predictions
        # return img from sam function
        img = sam_no_finetune_with_points(data)
        # plot the image
        axs[1].imshow(img)
        axs[1].set_title("sam_no_finetune_with_points")
        axs[1].axis("off")

        """
        # plot second figure with maskrcnn predictions
        # return img from maskrcnn function
        img = maskrcnn(data)
        # plot the image
        axs[0, 1].imshow(img)
        axs[0, 1].set_title('maskrcnn')
        axs[0, 1].axis('off')

        # plot third figure with sam predictions
        # return img from sam function
        img = sam_no_finetune(data)
        # plot the image
        axs[0, 2].imshow(img)
        axs[0, 2].set_title('sam_no_finetune')
        axs[0, 2].axis('off')

        # plot fourth figure with sam predictions
        # return img from sam function
        img = sam_with_finetune(data)
        # plot the image
        axs[1, 0].imshow(img)
        axs[1, 0].set_title('sam_with_finetune')
        axs[1, 0].axis('off')

        # plot fifth figure with sam + maskrcnn predictions
        # return img from sam function
        img = sam_maskrcnn_without_finetune(data)
        # plot the image
        axs[1, 1].imshow(img)
        axs[1, 1].set_title('sam_maskrcnn_without_finetune')
        axs[1, 1].axis('off')

        # plot sixth figure with sam + maskrcnn predictions
        # return img from sam function
        img = sam_maskrcnn_with_finetune(data)
        # plot the image
        axs[1, 2].imshow(img)
        axs[1, 2].set_title('sam_maskrcnn_with_finetune')
        axs[1, 2].axis('off')
        """

        # tight layout
        plt.tight_layout()

        # save the plot
        plt.savefig(
            "data/{}/predictions/{}.png".format(
                args.dataset, data["file_name"].split("/")[-1].split(".")[0]
            )
        )
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset")
    args = parser.parse_args()
    generate_predictions(args)
