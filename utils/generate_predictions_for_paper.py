#!/usr/bin/env python
"""
this script is used to generate predictions for all models in experiments
The output file will be saved in predictions_for_paper folder
For each image, there will be 11 output files:
    - image_name.png: the original image
    - image_name_gt.png: the ground truth mask
    - image_name_sam_grid.png: the predicted mask using sam with grid points
    - image_name_sam_gtbbox.png: the predicted mask using sam with ground truth bboxes
    - image_name_sam_gtpoint.png: the predicted mask using sam with ground truth points
    - image_name_sam_bbox.png: the predicted mask using sam with predicted bboxes
    - image_name_sam_finetune_gtbbox.png: the predicted mask using sam with finetune and ground truth bboxes
    - image_name_sam_finetune_gtpoint.png: the predicted mask using sam with finetune and ground truth points
    - image_name_sam_finetune_bbox.png: the predicted mask using sam with finetune and predicted bboxes
    - image_name_maskrcnn.png: the predicted mask using maskrcnn
    - image_name_sam_clip.png: the predicted mask using sam with clip
    - combined_image_name.png: the combined image with all the above images
"""

import argparse
import json
import random
from detectron2.data import DatasetCatalog
import cv2
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from segment_anything import SamPredictor, sam_model_registry


class customVisualizer(Visualizer):
    def __init__(
        self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE
    ):
        super().__init__(img_rgb, metadata, scale, instance_mode)

    def overlay_instances(self, masks=None, edge_color=None, alpha=0.7):
        masks = self._convert_masks(masks)
        num_instances = len(masks)
        colors = []
        for _ in range(num_instances):
            colors.append([random.randint(127, 255) for _ in range(3)])
            colors[-1] = [x / 255 for x in colors[-1]]
        areas = None
        areas = np.asarray([x.area() for x in masks])
        sorted_idxs = np.argsort(-areas).tolist()
        masks = [masks[i] for i in sorted_idxs]
        colors = [colors[i] for i in sorted_idxs]

        for i in range(num_instances):
            color = colors[i]
            for segment in masks[i].polygons:
                self.draw_polygon(
                    segment.reshape(-1, 2),
                    color=color,
                    edge_color=edge_color,
                    alpha=alpha,
                )
        return self.output
    
    def draw_polygon(self, segment, color, edge_color=None, alpha=0.7):
        edge_color = mplc.to_rgb(edge_color) + (1,)
        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor = edge_color,
            linewidth=2,
        )
        self.output.ax.add_patch(polygon)
        return self.output


def plot_image(axs, img, title):
    # the image format is BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axs.imshow(img)
    axs.set_title(title)
    axs.axis("off")


def plot_mask(img, masks, edge_color, alpha=0.7):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    v = customVisualizer(img)
    v = v.overlay_instances(masks=masks, edge_color=edge_color, alpha=alpha)
    img = v.get_image()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def save_image(img, args, name, title):
    # save the image
    cv2.imwrite(
        "predictions_for_paper/{}/{}_{}.png".format(args.dataset, name, title),
        img,
    )


def maskrcnn(data, args):
    # hyperparameters for prediction
    # dataset
    args.things_classes = [args.dataset]  # classes to be detected
    args.config_file = "config_files/mask_rcnn_R_50_FPN_3x.yaml"
    args.num_classes = 1
    args.seed = 1129
    if args.dataset == "iwp":
        args.model_weight = "experiments/maskrcnn/results/{}/model_{}.pth".format(
            args.dataset, "0001199"
        )
        args.anchor_sizes = [[32, 64, 128, 256, 512]]
    elif args.dataset == "rts":
        args.model_weight = "experiments/maskrcnn/results/rts/model_2nd_0001049.pth"
        args.anchor_sizes = [[16, 32, 64, 128, 256]]
    # model with default config
    args.min_size_train = (640, 672, 704, 736, 768, 800)
    args.max_size_train = 1333
    args.min_size_test = 800
    args.max_size_test = 1333
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
    cfg.MODEL.DEVICE = args.device

    # maskrcnn predictor
    predictor = DefaultPredictor(cfg)
    # predict
    img = cv2.imread(data["file_name"])
    outputs = predictor(img)

    scores = outputs["instances"].scores.cpu().numpy()
    masks = []
    # iterate through all masks
    for i in range(outputs["instances"].pred_masks.shape[0]):
        if scores[i] < 0.5:
            continue
        mask = outputs["instances"].pred_masks[i].cpu().numpy()
        masks.append(mask)

    img = plot_mask(img, masks, edge_color='b')
    return img

def get_data_dicts(dataset, split):
    import json 
    dataset_dics = json.load(open("data/{}/{}_{}.json".format(dataset, dataset, split)))
    return dataset_dics

def sam_mvitv2_without_finetune(sample, args):
    from detectron2.config import LazyConfig, instantiate
    from detectron2.engine import default_setup
    from detectron2.data import DatasetCatalog
    import torch 
   
    args.config_file = "config_files/cascade_mask_rcnn_mvitv2_b_in21k_3x.py"
    args.seed = 1129
    args.num_gpus = 1
    args.num_classes = 1
    args.batch_size = 1
    args.output_dir = 'predictions_for_paper/tmp/{}'.format(args.dataset)
    
    if args.dataset == "iwp":
        args.box_checkpoint = "experiments/mvitv2/results/iwp/mvitv2_b_in21k/model_0002599.pth"
        args.mask_checkpoint = "experiments/mvitv2/results/iwp/mvitv2_b_in21k/model_0002199.pth"
        args.anchor_sizes = [[32], [64], [128], [256], [512]]
        args.detections_per_img = 500
    elif args.dataset == "rts":
        args.box_checkpoint = "experiments/mvitv2/results/rts/mvitv2_b_in21k/model_0004599.pth"
        args.mask_checkpoint = "experiments/mvitv2/results/rts/mvitv2_b_in21k/model_0005999.pth"
        args.anchor_sizes = [[16], [32], [64], [128], [256]]
        args.detections_per_img = 100 

    cfg = LazyConfig.load(args.config_file)
    cfg.model.proposal_generator.anchor_generator.sizes = args.anchor_sizes
    cfg.model.roi_heads.num_classes = args.num_classes
    cfg.train.output_dir = args.output_dir
    cfg.train.seed = args.seed 
    cfg.dataloader.test.dataset.names  = args.dataset + "_test"
    cfg.dataloader.train.total_batch_size = args.batch_size
    cfg.dataloader.evaluator.max_dets_per_image = args.detections_per_img
    cfg.dataloader.evaluator.output_dir = args.output_dir

    default_setup(cfg, args)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    DetectionCheckpointer(model).load(args.box_checkpoint)
    model.eval()

    test_data_loader = instantiate(cfg.dataloader.test)
    for idx, data in enumerate(test_data_loader):
        if data[0]['file_name'] != sample["file_name"]:
            continue

        with torch.no_grad():
            output = model(data)

        # sam predictor with finetune 
        sam = sam_model_registry["vit_h"](checkpoint="pretrained_weights/sam_vit_h_4b8939.pth")
        sam.to(cfg.train.device)
        sam.eval()
        predictor = SamPredictor(sam)

        img = cv2.imread(data[0]["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(img)

        masks = []
        scores = output[0]["instances"].scores
        boxes = output[0]["instances"].pred_boxes.tensor

        for i in range(boxes.shape[0]):
            if scores[i] < 0.5:
                continue
            box = boxes[i].cpu().numpy()
            box = box.astype(np.int32)
            mask, score, _ = predictor.predict(box=box, multimask_output=False)
            mask = mask.squeeze(0)
            masks.append(mask)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = plot_mask(img, masks, edge_color='b')
        return img 


def sam_mvitv2_with_finetune(sample, args):
    from detectron2.config import LazyConfig, instantiate
    from detectron2.engine import default_setup
    from detectron2.data import DatasetCatalog
    import torch 

    args.config_file = "config_files/cascade_mask_rcnn_mvitv2_b_in21k_3x.py"
    args.seed = 1129
    args.num_gpus = 1
    args.num_classes = 1
    args.batch_size = 1
    args.output_dir = 'predictions_for_paper/tmp/{}'.format(args.dataset)
    
    if args.dataset == "iwp":
        args.box_checkpoint = "experiments/mvitv2/results/iwp/mvitv2_b_in21k/model_0002599.pth"
        args.mask_checkpoint = "experiments/mvitv2/results/iwp/mvitv2_b_in21k/model_0002199.pth"
        args.anchor_sizes = [[32], [64], [128], [256], [512]]
        args.detections_per_img = 500
    elif args.dataset == "rts":
        args.box_checkpoint = "experiments/mvitv2/results/rts/mvitv2_b_in21k/model_0004599.pth"
        args.mask_checkpoint = "experiments/mvitv2/results/rts/mvitv2_b_in21k/model_0005999.pth"
        args.anchor_sizes = [[16], [32], [64], [128], [256]]
        args.detections_per_img = 100 

    cfg = LazyConfig.load(args.config_file)
    cfg.model.proposal_generator.anchor_generator.sizes = args.anchor_sizes
    cfg.model.roi_heads.num_classes = args.num_classes
    cfg.train.output_dir = args.output_dir
    cfg.train.seed = args.seed 
    cfg.dataloader.test.dataset.names  = args.dataset + "_test"
    cfg.dataloader.train.total_batch_size = args.batch_size
    cfg.dataloader.evaluator.max_dets_per_image = args.detections_per_img
    cfg.dataloader.evaluator.output_dir = args.output_dir

    default_setup(cfg, args)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    DetectionCheckpointer(model).load(args.box_checkpoint)
    model.eval()

    test_data_loader = instantiate(cfg.dataloader.test)
    for idx, data in enumerate(test_data_loader):
        if data[0]['file_name'] != sample["file_name"]:
            continue

        with torch.no_grad():
            output = model(data)

        # sam predictor with finetune 
        sam = sam_model_registry["vit_h"]()
        if args.dataset == "iwp":
            model_weights = (
                "experiments/sam_with_finetune/results/iwp/bbox/model_0000139.pth"
            )
        elif args.dataset == "rts":
            model_weights = (
                "experiments/sam_with_finetune/results/rts/bbox/model_0000149.pth"
            )
        else:
            raise NotImplementedError
        DetectionCheckpointer(sam).load(model_weights)
        sam.to(cfg.train.device)
        sam.eval()
        predictor = SamPredictor(sam)

        img = cv2.imread(data[0]["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(img)

        masks = []
        scores = output[0]["instances"].scores
        boxes = output[0]["instances"].pred_boxes.tensor

        for i in range(boxes.shape[0]):
            if scores[i] < 0.5:
                continue
            box = boxes[i].cpu().numpy()
            box = box.astype(np.int32)
            mask, score, _ = predictor.predict(box=box, multimask_output=False)
            mask = mask.squeeze(0)
            masks.append(mask)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = plot_mask(img, masks, edge_color='b')
        return img 
    

def mvitv2(sample, args):
    from detectron2.config import LazyConfig, instantiate
    from detectron2.engine import default_setup
    from detectron2.data import DatasetCatalog
    import torch 

    args.config_file = "config_files/cascade_mask_rcnn_mvitv2_b_in21k_3x.py"
    args.seed = 1129
    args.num_gpus = 1
    args.num_classes = 1
    args.batch_size = 1
    args.output_dir = 'predictions_for_paper/tmp/{}'.format(args.dataset)
    
    if args.dataset == "iwp":
        args.box_checkpoint = "experiments/mvitv2/results/iwp/mvitv2_b_in21k/model_0002599.pth"
        args.mask_checkpoint = "experiments/mvitv2/results/iwp/mvitv2_b_in21k/model_0002199.pth"
        args.anchor_sizes = [[32], [64], [128], [256], [512]]
        args.detections_per_img = 500
    elif args.dataset == "rts":
        args.box_checkpoint = "experiments/mvitv2/results/rts/mvitv2_b_in21k/model_0004599.pth"
        args.mask_checkpoint = "experiments/mvitv2/results/rts/mvitv2_b_in21k/model_0005999.pth"
        args.anchor_sizes = [[16], [32], [64], [128], [256]]
        args.detections_per_img = 100 

    cfg = LazyConfig.load(args.config_file)
    cfg.model.proposal_generator.anchor_generator.sizes = args.anchor_sizes
    cfg.model.roi_heads.num_classes = args.num_classes
    cfg.train.output_dir = args.output_dir
    cfg.train.seed = args.seed 
    cfg.dataloader.test.dataset.names  = args.dataset + "_test"
    cfg.dataloader.train.total_batch_size = args.batch_size
    cfg.dataloader.evaluator.max_dets_per_image = args.detections_per_img
    cfg.dataloader.evaluator.output_dir = args.output_dir

    default_setup(cfg, args)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    DetectionCheckpointer(model).load(args.box_checkpoint)
    model.eval()

    test_data_loader = instantiate(cfg.dataloader.test)
    for idx, data in enumerate(test_data_loader):
        if data[0]['file_name'] != sample["file_name"]:
            continue

        with torch.no_grad():
            output = model(data)

        img = cv2.imread(data[0]["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        scores = output[0]["instances"].scores.cpu().numpy()
        masks = []
        # iterate through all masks
        for i in range(output[0]["instances"].pred_masks.shape[0]):
            if scores[i] < 0.5:
                continue
            mask = output[0]["instances"].pred_masks[i].cpu().numpy()
            masks.append(mask)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = plot_mask(img, masks, edge_color='b')
        return img 
    

def sam_no_finetune(data, prompt="bbox"):
    # use ground truth box/point to run sam
    # load image
    img = cv2.imread(data["file_name"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # load sam
    sam = sam_model_registry["vit_h"](
        checkpoint="pretrained_weights/sam_vit_h_4b8939.pth"
    )
    sam.to("cuda:0")
    sam.eval()
    predictor = SamPredictor(sam)

    # load image into sam
    predictor.set_image(img)

    masks = []
    # iterate through all annotations
    for anno in data["annotations"]:
        if prompt == "bbox":
            box = anno["bbox"]
            box = np.array(box).astype(np.int32)
            # run sam
            mask, score, _ = predictor.predict(box=box, multimask_output=False)
        elif prompt == "point":
            x1, y1, x2, y2 = anno["bbox"]
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            point_coords = np.array([x, y]).astype(np.int32).reshape(1, 2)
            point_labels = np.array([1]).astype(np.int32)
            mask, score, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
        else:
            raise NotImplementedError
        mask = mask.squeeze(0)
        masks.append(mask)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = plot_mask(img, masks, edge_color='b')
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


def sam_with_finetune(data, args, prompt="bbox"):
    # use ground truth boxes to run sam
    # load sam model
    sam = sam_model_registry["vit_h"]()

    if args.dataset == "iwp" and prompt == "bbox":
        model_weights = (
            "experiments/sam_with_finetune/results/iwp/bbox/model_0000139.pth"
        )
    elif args.dataset == "iwp" and prompt == "point":
        model_weights = (
            "experiments/sam_with_finetune/results/iwp/point/model_0000169.pth"
        )
    elif args.dataset == "rts" and prompt == "bbox":
        model_weights = (
            "experiments/sam_with_finetune/results/rts/bbox/model_0000149.pth"
        )
    elif args.dataset == "rts" and prompt == "point":
        model_weights = (
            "experiments/sam_with_finetune/results/rts/point/model_0000149.pth"
        )
    else:
        raise NotImplementedError

    DetectionCheckpointer(sam).load(model_weights)
    sam.to("cuda:0")
    sam.eval()
    predictor = SamPredictor(sam)
    # load image into sam
    img = cv2.imread(data["file_name"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img)

    masks = []
    # iterate through all annotations
    for anno in data["annotations"]:
        if prompt == "bbox":
            box = anno["bbox"]
            box = np.array(box).astype(np.int32)
            # run sam
            mask, score, _ = predictor.predict(box=box, multimask_output=False)
        elif prompt == "point":
            x1, y1, x2, y2 = anno["bbox"]
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            point_coords = np.array([x, y]).astype(np.int32).reshape(1, 2)
            point_labels = np.array([1]).astype(np.int32)
            mask, score, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
        else:
            raise NotImplementedError
        mask = mask.squeeze(0)
        masks.append(mask)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = plot_mask(img, masks, edge_color='b')

    return img


def sam_maskrcnn_without_finetune(data, args):
    # dataset
    args.things_classes = [args.dataset]  # classes to be detected
    args.config_file = "config_files/mask_rcnn_R_50_FPN_3x.yaml"
    args.num_classes = 1
    args.seed = 1129
    if args.dataset == "iwp":
        args.model_weight = "experiments/maskrcnn/results/{}/model_{}.pth".format(
            args.dataset, "0001199"
        )
        args.anchor_sizes = [[32, 64, 128, 256, 512]]
    elif args.dataset == "rts":
        args.model_weight = "experiments/maskrcnn/results/rts/model_2nd_0001049.pth"
        args.anchor_sizes = [[16, 32, 64, 128, 256]]
    # model with default config
    args.min_size_train = (640, 672, 704, 736, 768, 800)
    args.max_size_train = 1333
    args.min_size_test = 800
    args.max_size_test = 1333
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
    cfg.MODEL.DEVICE = args.device

    # maskrcnn predictor
    predictor = DefaultPredictor(cfg)
    # predict
    img = cv2.imread(data["file_name"])
    outputs = predictor(img)

    # sam predictor without finetune
    sam = sam_model_registry["vit_h"](
        checkpoint="pretrained_weights/sam_vit_h_4b8939.pth"
    )
    predictor = SamPredictor(sam)

    # load image into sam with rgb format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img)
    # maskrcnn box output
    scores = outputs["instances"].scores
    boxes = outputs["instances"].pred_boxes.tensor
    # iterate through all annotations
    masks = []
    for i in range(boxes.shape[0]):
        if scores[i] < 0.5:
            continue
        box = boxes[i].cpu().numpy()
        box = box.astype(np.int32)
        # run sam
        mask, score, _ = predictor.predict(box=box, multimask_output=False)
        mask = mask.squeeze(0)
        masks.append(mask)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = plot_mask(img, masks, edge_color='b')
    return img


def sam_maskrcnn_with_finetune(data, args):
    # hyperparameters for prediction
    # dataset
    args.things_classes = [args.dataset]  # classes to be detected
    args.config_file = "config_files/mask_rcnn_R_50_FPN_3x.yaml"
    args.num_classes = 1
    args.seed = 1129
    if args.dataset == "iwp":
        args.model_weight = "experiments/maskrcnn/results/{}/model_{}.pth".format(
            args.dataset, "0001199"
        )
        args.anchor_sizes = [[32, 64, 128, 256, 512]]
    elif args.dataset == "rts":
        args.model_weight = "experiments/maskrcnn/results/rts/model_2nd_0001049.pth"
        args.anchor_sizes = [[16, 32, 64, 128, 256]]
    # model with default config
    args.min_size_train = (640, 672, 704, 736, 768, 800)
    args.max_size_train = 1333
    args.min_size_test = 800
    args.max_size_test = 1333
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
    cfg.MODEL.DEVICE = args.device

    # maskrcnn predictor
    predictor = DefaultPredictor(cfg)
    # predict
    img = cv2.imread(data["file_name"])
    outputs = predictor(img)

    # sam predictor with finetune
    sam = sam_model_registry["vit_h"]()
    if args.dataset == "iwp":
        model_weights = (
            "experiments/sam_with_finetune/results/iwp/bbox/model_0000139.pth"
        )
    elif args.dataset == "rts":
        model_weights = (
            "experiments/sam_with_finetune/results/rts/bbox/model_0000149.pth"
        )
    else:
        raise NotImplementedError
    sam.to("cuda:0")
    sam.eval()
    DetectionCheckpointer(sam).load(model_weights)
    predictor = SamPredictor(sam)

    # load image into sam with rgb format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img)
    # maskrcnn box output
    scores = outputs["instances"].scores
    boxes = outputs["instances"].pred_boxes.tensor
    # iterate through all annotations
    masks = []
    for i in range(boxes.shape[0]):
        if scores[i] < 0.5:
            continue
        box = boxes[i].cpu().numpy()
        box = box.astype(np.int32)
        # run sam
        mask, score, _ = predictor.predict(box=box, multimask_output=False)
        mask = mask.squeeze(0)
        masks.append(mask)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = plot_mask(img, masks, edge_color='b')
    return img


def generate_predictions(args):
    # load test json file
    test_json = json.load(
        open("data/{}/{}_test.json".format(args.dataset, args.dataset))
    )

    # iterate through all data in the test json file
    for idx, data in enumerate(test_json):
        if args.file is not None and data["file_name"].split("/")[-1] != args.file:
            continue

        # print progress
        print("processing {}/{}".format(idx, len(test_json)))

        # plot a 4x3 figure for each image
        fig, axs = plt.subplots(4, 3, figsize=(15, 20))

        # image path and image name
        img_path = data["file_name"]
        img_name = img_path.split("/")[-1].split(".")[0]
        img_original = cv2.imread(img_path)  # in bgr format

        # plot first figure - the original image
        img = img_original.copy()
        plot_image(axs[0, 0], img, "original image")
        save_image(img, args, img_name, "original")

        # plot second figure - the original image with ground truth
        img = img_original.copy()
        masks = []
        # load all masks from the ground truth
        for anno in data["annotations"]:
            masks.append(anno["segmentation"])
        img = plot_mask(img, masks, edge_color='r')
        plot_image(axs[0, 1], img, "ground truth")
        save_image(img, args, img_name, "gt")

        # plot third figure - sam predictions using grid points
        img_name_in_rts = img_name
        if args.dataset == "rts":
            # extract the number of the image
            img_name_in_rts = img_name.split("_")[-1]
        img = cv2.imread(
            "/data/shared/sam_expr_result/{}/sam/{}.png".format(
                args.dataset, img_name_in_rts
            )
        )
        plot_image(axs[0, 2], img, "sam with grid points")
        save_image(img, args, img_name, "sam_grid")

        # plot fourth figure - sam predictions using ground truth bbox
        img = sam_no_finetune(data, prompt="bbox")
        plot_image(axs[1, 0], img, "sam with ground truth bbox")
        save_image(img, args, img_name, "sam_gt_bbox")

        # plot fifth figure - sam predictions using ground truth point
        img = sam_no_finetune(data, prompt="point")
        plot_image(axs[1, 1], img, "sam with ground truth point")
        save_image(img, args, img_name, "sam_gt_point")

        # plot sixth figure - sam predictions using predicted bbox
        img = sam_mvitv2_without_finetune(data, args)
        plot_image(axs[1, 2], img, "sam with predicted bbox")
        save_image(img, args, img_name, "sam_pred_bbox")

        # plot seventh figure - sam predictions with finetune using ground truth bbox
        img = sam_with_finetune(data, args, prompt="bbox")
        plot_image(axs[2, 0], img, "sam with finetune and ground truth bbox")
        save_image(img, args, img_name, "sam_finetune_gt_bbox")

        # plot eighth figure - sam predictions with finetune using ground truth point
        img = sam_with_finetune(data, args, prompt="point")
        plot_image(axs[2, 1], img, "sam with finetune and ground truth point")
        save_image(img, args, img_name, "sam_finetune_gt_point")

        # plot ninth figure - sam predictions with finetune using predicted bbox
        img = sam_mvitv2_with_finetune(data, args)
        plot_image(axs[2, 2], img, "sam with finetune and predicted bbox")
        save_image(img, args, img_name, "sam_finetune_pred_bbox")

        # plot tenth figure - predictions using maskrcnn
        # img = maskrcnn(data, args)
        # plot_image(axs[3, 0], img, "maskrcnn")
        # save_image(img, args, img_name, "maskrcnn")
        img = mvitv2(data, args)
        plot_image(axs[3, 0], img, "mvitv2")   
        save_image(img, args, img_name, "mvitv2")

        # plot eleventh figure - predictions using sam with clip
        img_name_in_rts = img_name
        if args.dataset == "rts":
            img_name_in_rts = img_name.split("_")[-1]   
        img = cv2.imread(
            "/data/shared/sam_expr_result/{}/pred/{}.png".format(
                args.dataset, img_name_in_rts
            )
        )
        plot_image(axs[3, 1], img, "sam with clip")
        save_image(img, args, img_name, "sam_clip")


        # tight layout
        plt.tight_layout()
        # save the figure to predictions_for_paper/args.dataset/combined_file_name.png
        plt.savefig(
            "predictions_for_paper/{}/combined_{}.png".format(args.dataset, img_name)
        )
        # close the figure
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--file", type=str)
    args = parser.parse_args()

    for d in ["train", "test"]:
        DatasetCatalog.register(
            args.dataset + "_" + d, lambda d=d: get_data_dicts(args.dataset, d)
        )

    generate_predictions(args)
