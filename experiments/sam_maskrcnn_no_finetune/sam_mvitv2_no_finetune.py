#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging

import cv2
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import (AMPTrainer, SimpleTrainer,
                               default_argument_parser, default_setup,
                               default_writers, hooks, launch)
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

logger = logging.getLogger("detectron2")


def maskrcnn_eval(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def sam_maskrcnn_eval(cfg, maskrcnn):
    maskrcnn.eval()
    test_data_loader = instantiate(cfg.dataloader.test)
    evaluator = instantiate(cfg.dataloader.evaluator)
    evaluator.reset()

    sam = sam_model_registry["vit_h"](
        checkpoint="pretrained_weights/sam_vit_h_4b8939.pth"
    )
    sam.to(cfg.train.device)
    sam.eval()
    transform = ResizeLongestSide(sam.image_encoder.img_size) 
    for idx, inputs in enumerate(test_data_loader):
        # print progress every 10 images
        if idx % 10 == 0:
            print("processing {}/{}".format(idx, len(test_data_loader)))
        with torch.no_grad():
            maskrcnn_output = maskrcnn(inputs)

        # create input for SAM
        sam_input = [{}]
        img = cv2.imread(inputs[0]["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sam_input[0]["original_size"] = (img.shape[0], img.shape[1])
        img = transform.apply_image(img)
        sam_input[0]["image"] = torch.as_tensor(img.astype("float32")).permute(2, 0, 1).to(cfg.train.device)
        boxes = maskrcnn_output[0]["instances"].pred_boxes.tensor.detach().cpu().numpy()
        boxes = transform.apply_boxes(boxes, sam_input[0]["original_size"])
        sam_input[0]["boxes"] = torch.as_tensor(boxes).to(cfg.train.device)

        with torch.no_grad():
            sam_output = sam(sam_input, multimask_output=False, required_grad=False)

        maskrcnn_output[0]["instances"].pred_masks = sam_output[0]["masks"].squeeze(1)
        evaluator.process(inputs, maskrcnn_output)

    results = evaluator.evaluate()
     
def get_data_dicts(dataset, split):
    import json 
    dataset_dics = json.load(open("data/{}/{}_{}.json".format(dataset, dataset, split)))
    return dataset_dics


def main(args):
    # register dataset
    for d in ["train", "test"]:
        DatasetCatalog.register(
            args.dataset + "_" + d, lambda d=d: get_data_dicts(args.dataset, d)
        )
        MetadataCatalog.get(args.dataset + "_" + d).set(thing_classes=args.things_classes)


    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    # key: model, optimizer, train, lr_multiplier, dataloader
    # customize model parameters
    cfg.model.proposal_generator.anchor_generator.sizes = args.anchor_sizes
    cfg.model.roi_heads.num_classes = args.num_classes

    # customize optimizer parameters
    # customize train parameters
    cfg.train.output_dir = args.output_dir
    cfg.train.seed = args.seed 

    # customize lr_multiplier parameters
    # customize dataloader parameters
    cfg.dataloader.train.dataset.names = args.dataset + "_train"
    cfg.dataloader.test.dataset.names  = args.dataset + "_test"
    cfg.dataloader.evaluator.max_dets_per_image = args.detections_per_img
    cfg.dataloader.evaluator.output_dir = args.output_dir

    default_setup(cfg, args)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    for mode in ["mask", "box"]:
        DetectionCheckpointer(model).load(args.box_checkpoint if mode == "box" else args.mask_checkpoint)
        print("evaluating {}...".format(mode))
        print(maskrcnn_eval(cfg, model))
        
    sam_maskrcnn_eval(cfg, model)

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--dataset", type=str, default="dataset")
    args = parser.parse_args()

    args.things_classes = [args.dataset]
    args.num_classes = 1
    # use mvitv2_in21k as backbone
    args.config_file = "config_files/cascade_mask_rcnn_mvitv2_b_in21k_3x.py"
    args.seed = 1129
    args.output_dir = 'experiments/sam_maskrcnn_no_finetune/results/{}/mvitv2_b_in21k'.format(args.dataset)
    
    
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
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
