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

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.data import DatasetCatalog, MetadataCatalog

logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))

    # calculate number of parameters
    total_params = sum(param.numel() for param in model.parameters())
    logger.info("Number of parameters: {}".format(total_params))

    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(
        model, train_loader, optim
    )
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            (
                hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
                if comm.is_main_process()
                else None
            ),
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            (
                hooks.PeriodicWriter(
                    default_writers(cfg.train.output_dir, cfg.train.max_iter),
                    period=cfg.train.log_period,
                )
                if comm.is_main_process()
                else None
            ),
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


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
        MetadataCatalog.get(args.dataset + "_" + d).set(
            thing_classes=args.things_classes
        )

    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    # key: model, optimizer, train, lr_multiplier, dataloader
    # customize model parameters
    cfg.model.proposal_generator.anchor_generator.sizes = args.anchor_sizes
    cfg.model.roi_heads.num_classes = args.num_classes

    # customize optimizer parameters
    # customize train parameters
    cfg.train.output_dir = args.output_dir
    cfg.train.init_checkpoint = args.init_checkpoint
    cfg.train.max_iter = (
        args.epoch * len(DatasetCatalog.get(args.dataset + "_train")) // args.batch_size
    )
    cfg.train.checkpointer.period = args.checkpoint_period
    cfg.train.eval_period = args.eval_period
    cfg.train.seed = args.seed

    # customize lr_multiplier parameters
    # customize dataloader parameters
    cfg.dataloader.train.dataset.names = args.dataset + "_train"
    cfg.dataloader.test.dataset.names = args.dataset + "_test"
    cfg.dataloader.train.total_batch_size = args.batch_size
    cfg.dataloader.evaluator.max_dets_per_image = args.detections_per_img
    cfg.dataloader.evaluator.output_dir = args.output_dir

    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--dataset", type=str, default="dataset")
    args = parser.parse_args()

    args.things_classes = [args.dataset]
    args.num_classes = 1
    args.config_file = "config_files/cascade_mask_rcnn_mvitv2_b_in21k_3x.py"
    args.init_checkpoint = "pretrained_weights/mvitv2_b_cascade_mask_rcnn.pkl"
    args.seed = 1129
    args.num_gpus = 4
    args.batch_size = 4
    args.epoch = 100
    args.checkpoint_period = 200
    args.eval_period = 200
    args.output_dir = "experiments/mvitv2/results/{}/mvitv2_b_in21k".format(
        args.dataset
    )

    if args.dataset == "iwp":
        args.anchor_sizes = [[32], [64], [128], [256], [512]]
        args.detections_per_img = 500
    elif args.dataset == "rts":
        args.anchor_sizes = [[16], [32], [64], [128], [256]]
        args.detections_per_img = 100
    elif args.dataset == "agr":
        args.anchor_sizes = [[16], [32], [64], [128], [256]]
        args.detections_per_img = 100
    else:
        raise ValueError("Invalid dataset name")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
