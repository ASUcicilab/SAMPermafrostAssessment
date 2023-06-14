#!/usr/bin/env python

"""
This script is to evalute SAM's segmentation quality after finetuning on the train set
The ground truth bounding box is used as prompt.
"""

import argparse
import copy
import datetime
import json
import logging
import os
import sys
import time
import weakref
from collections import OrderedDict
from contextlib import ExitStack

import numpy as np
import torch
import torch.nn as nn
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data import detection_utils as utils
from detectron2.engine import (
    DefaultTrainer,
    SimpleTrainer,
    create_ddp_model,
    default_setup,
    launch,
)
from detectron2.evaluation import COCOEvaluator, inference_context, print_csv_format
from detectron2.structures import Instances
from detectron2.structures.masks import polygons_to_bitmask
from detectron2.utils import comm
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import log_every_n_seconds
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


def customMapper(dataset_dict, training=True, prompt="bbox"):
    """
    Customized mapper for SAM finetuning.
    batched_input (list(dict)): A list over input images, each a dictionary with
    the following keys. A prompt key can be excluded if it is not present.
        'image': The image as a torch tensor in 3xHxW format, already transformed for input to the model.
        'original_size': (tuple(int, int)) The original size of the image before transformation, as (H, W).
        'point_coords': (torch.Tensor) Batched point prompts for this image, with shape BxNx2.
            Already transformed to the input frame of the model.
        'point_labels': (torch.Tensor) Batched labels for point prompts, with shape BxN.
        'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
            Already transformed to the input frame of the model.
    """

    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="RGB")
    utils.check_image_size(dataset_dict, image)

    if training:
        # take the first 100 annotations or less to fit into gpu
        dataset_dict["annotations"] = dataset_dict["annotations"][:100]

    # collect bbox, category_id and segmentation from annotations
    point_coords = []
    bboxes = []
    for anno in dataset_dict["annotations"]:
        bboxes.append(anno["bbox"])
        x1, y1, x2, y2 = anno["bbox"]
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        point_coords.append([x, y])

    instances = utils.annotations_to_instances(
        dataset_dict["annotations"], image.shape[:2]
    )
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    # transform the image
    transform = ResizeLongestSide(1024)
    image = transform.apply_image(image)
    dataset_dict["image"] = torch.as_tensor(
        np.ascontiguousarray(image.transpose(2, 0, 1))
    )
    dataset_dict["original_size"] = (dataset_dict["height"], dataset_dict["width"])

    # transform the bbox, category_id and segmentation to torch tensor
    if prompt == "bbox":
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        dataset_dict["boxes"] = transform.apply_boxes_torch(
            bboxes, dataset_dict["original_size"]
        )
    elif prompt == "point":
        point_coords = np.asarray(point_coords)
        point_coords = transform.apply_coords(
            point_coords, original_size=dataset_dict["original_size"]
        )
        point_coords = torch.as_tensor(point_coords, dtype=torch.float32).unsqueeze(1)
        dataset_dict["point_coords"] = point_coords
        dataset_dict["point_labels"] = torch.ones(len(point_coords), 1)
        
    dataset_dict.pop("annotations", None)

    return dataset_dict


def custom_inference_on_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            # skip empty images
            if len(inputs[0]["instances"]) == 0:
                continue

            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs, multimask_output=False, required_grad=False)

            # what's the input and output format?
            transformed_outputs = []
            transformed_outputs.append(
                {"instances": Instances(image_size=inputs[0]["instances"].image_size)}
            )
            transformed_outputs[0]["instances"].set(
                "pred_masks", outputs[0]["masks"].squeeze(1)
            )
            transformed_outputs[0]["instances"].set(
                "pred_boxes", inputs[0]["instances"].gt_boxes
            )
            transformed_outputs[0]["instances"].set(
                "pred_classes", inputs[0]["instances"].gt_classes
            )
            transformed_outputs[0]["instances"].set(
                "scores", outputs[0]["iou_predictions"].squeeze(1)
            )

            outputs = transformed_outputs

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (
                time.perf_counter() - start_time
            ) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(
                    seconds=int(total_seconds_per_iter * (total - idx - 1))
                )
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total - num_warmup),
            num_devices,
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


class CustomSimpleTrainer(SimpleTrainer):
    """
    customized SimpleTrainer for SAM
    """

    def dice_loss(self, prediction, target):
        """
        calculate the dice loss
        """
        smooth = 1e-5
        pflat = prediction.reshape(1, -1)
        tflat = target.reshape(1, -1)
        intersection = (pflat * tflat).sum()
        total = pflat.sum() + tflat.sum()
        dice_score = (2.0 * intersection + smooth) / (total + smooth)
        dice_loss = 1 - dice_score
        return dice_loss

    def run_step(self):
        """
        Implement the custom training logic for SAM
        most of them are the same, besides the input of the model
        """

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        outputs = self.model(data, multimask_output=False, required_grad=True)

        # calculate the mask loss
        loss_dict = torch.zeros(1, requires_grad=True).to(self.model.device)
        for idx, output in enumerate(outputs):
            pred_masks = output["masks"]
            gt_masks = data[idx]["instances"].gt_masks.polygons
            gt_masks = [
                polygons_to_bitmask(polygon, data[idx]["height"], data[idx]["width"])
                for polygon in gt_masks
            ]
            # remove the second dimension of pred_masks
            pred_masks = pred_masks.squeeze(1)
            # combine all masks in the gt_masks list into one numpy array
            gt_masks = np.stack(gt_masks, axis=0)
            # convert the numpy array to torch tensor and to the same device as the pred_masks
            gt_masks = torch.as_tensor(
                gt_masks, dtype=torch.float32, device=pred_masks.device
            )
            # calculate the dice loss
            loss_dict += self.dice_loss(pred_masks, gt_masks)

        loss_dict /= len(outputs)

        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
        if not self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()

        losses.backward()
        self.after_backward()

        if self.async_write_metrics:
            # write metrics asynchronically
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        self.optimizer.step()


class Trainer(DefaultTrainer):
    """
    Customized trainer for SAM
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        model = self.build_model(cfg)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
        data_loader = build_detection_train_loader(
            cfg,
            mapper=lambda x: customMapper(x, training=True, prompt=cfg.prompt),
        )

        model = create_ddp_model(
            model, broadcast_buffers=False, find_unused_parameters=True
        )
        self._trainer = CustomSimpleTrainer(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )

        self._hooks = []
        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        """
        Building the SAM model
        """
        model = sam_model_registry["vit_h"](
            checkpoint="pretrained_weights/sam_vit_h_4b8939.pth"
        )
        model.to(torch.device(cfg.MODEL.DEVICE))
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = cfg.OUTPUT_DIR
        return COCOEvaluator(
            dataset_name,
            output_dir=output_folder,
            max_dets_per_image=cfg.TEST.DETECTIONS_PER_IMAGE,
        )

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Test the model
        """

        logger = logging.getLogger(__name__)
        evaluator = cls.build_evaluator(cfg, dataset_name=cfg.DATASETS.TEST[0])
        results = OrderedDict()
        data_loader = build_detection_test_loader(
            dataset=DatasetCatalog.get(cfg.DATASETS.TEST[0]),
            mapper=lambda x: customMapper(x, training=False, prompt=cfg.prompt),
        )
        results_i = custom_inference_on_dataset(model, data_loader, evaluator)
        results[cfg.DATASETS.TEST[0]] = results_i
        if comm.is_main_process():
            assert isinstance(
                results_i, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results_i
            )
            logger.info(
                "Evaluation results for {} in csv format:".format(cfg.DATASETS.TEST[0])
            )
            print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def get_data_dicts(dataset_name, split):
    """
    Load the data of the given dataset
    """
    dataset = json.load(
        open("data/{}/{}_{}.json".format(dataset_name, dataset_name, split))
    )
    return dataset


def setup(args):
    cfg = get_cfg()

    # register the training and testing dataset
    for d in ["train", "test"]:
        DatasetCatalog.register(
            args.dataset + "_" + d, lambda d=d: get_data_dicts(args.dataset, d)
        )
        MetadataCatalog.get(args.dataset + "_" + d).set(
            thing_classes=args.things_classes
        )

    # add project-specific config
    # add train, val, and test dataset
    cfg.DATASETS.TRAIN = (args.dataset + "_train",)
    cfg.DATASETS.TEST = (args.dataset + "_test",)
    cfg.DATASETS.VAL = (args.dataset + "_test",)
    cfg.MODEL.MASK_ON = True

    # training config
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.WEIGHT_DECAY = args.weight_decay
    cfg.SOLVER.MAX_ITER = (
        args.epoch * len(DatasetCatalog.get(args.dataset + "_train")) // args.batch_size
    )
    cfg.SEED = args.seed
    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.TEST.DETECTIONS_PER_IMAGE = args.max_dets_per_img
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.OUTPUT_DIR = args.output_dir
    cfg.prompt = args.prompt

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)

    return trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM finetuning with detectron2")
    parser.add_argument("--dataset", default="iwp", help="dataset name")
    parser.add_argument("--prompt", default="bbox", help="prompt type: bbox or point")

    args = parser.parse_args()

    # hyperparameters
    # dataset
    args.dataset = args.dataset
    args.things_classes = [args.dataset]
    args.num_classes = 1

    # training
    args.seed = 1129
    args.num_gpus = 4
    args.num_machines = 1
    args.machine_rank = 0
    args.base_lr = 0.0008
    args.weight_decay = 0.1
    args.batch_size = 4
    args.epoch = 100
    args.output_dir = "experiments/sam_with_finetune/results/{}/{}".format(
        args.dataset, args.prompt
    )
    args.port = (
        2**15
        + 2**14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    )
    args.dist_url = "tcp://127.0.0.1:{}".format(args.port)

    # customize the training
    if args.dataset == "iwp":
        args.max_dets_per_img = 500
        args.eval_period = 10
        args.checkpoint_period = 10
    elif args.dataset == "rts":
        args.max_dets_per_img = 100
        args.eval_period = 10
        args.checkpoint_period = 10
    else:
        raise NotImplementedError("dataset not supported")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
