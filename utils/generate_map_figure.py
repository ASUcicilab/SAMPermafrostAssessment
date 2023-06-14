#!/usr/bin/env python

"""
This script is used to generate the map figure for the specified dataset from the log file.
"""

import argparse

import matplotlib.pyplot as plt


def generate_map_figure(args):
    # check if the subexp exists
    if args.subexp:
        log_file = "experiments/{}/results/{}/{}/log.txt".format(
            args.experiment, args.dataset, args.subexp
        )
    else:
        log_file = "experiments/{}/results/{}/log.txt".format(
            args.experiment, args.dataset
        )

    with open(log_file) as f:
        # keep lines that contain 'copypaste' and without 'Task' or 'AP'
        lines = [
            line
            for line in f.readlines()
            if "copypaste" in line and "Task" not in line and "AP" not in line
        ]
        # keep contents after 'copypaste:'
        lines = [line.split("copypaste:")[1] for line in lines]
        # keep contents before '\n'
        lines = [line.split("\n")[0] for line in lines]
        # extract the second number of each line which is the map50
        lines = [line.split(",")[1] for line in lines]

        bbox_map50 = [float(line) for line in lines[0::2]]
        segm_map50 = [float(line) for line in lines[1::2]]

    # draw bbox map50 and segm map50 on the same figure
    # the x-axis is the iteration from args.checkpoint_period-1 and increment by args.checkpoint_period
    # the y-axis is the map50
    # highlight the max bbox map50 and segm map50 and label the iteration and the value
    plt.plot(
        range(args.period - 1, len(bbox_map50) * args.period, args.period),
        bbox_map50,
        label="bbox_map50",
    )
    plt.plot(
        range(args.period - 1, len(segm_map50) * args.period, args.period),
        segm_map50,
        label="segm_map50",
    )
    plt.scatter(
        (bbox_map50.index(max(bbox_map50)) + 1) * args.period - 1,
        max(bbox_map50),
        c="r",
        label="max_bbox_map50: {}, {:.2f}".format(
            (bbox_map50.index(max(bbox_map50)) + 1) * args.period - 1, max(bbox_map50)
        ),
    )
    plt.scatter(
        (segm_map50.index(max(segm_map50)) + 1) * args.period - 1,
        max(segm_map50),
        c="g",
        label="max_segm_map50: {}, {:.2f}".format(
            (segm_map50.index(max(segm_map50)) + 1) * args.period - 1, max(segm_map50)
        ),
    )
    plt.xlabel("iteration")
    plt.ylabel("map50")
    plt.legend()
    if args.subexp:
        plt.title(
            "map50 for {} on {} with {}".format(
                args.experiment, args.dataset, args.subexp
            )
        )
    else:
        plt.title("map50 for {} on {}".format(args.experiment, args.dataset))

    if args.subexp:
        plt.savefig(
            "experiments/{}/results/{}/{}/map_fig.png".format(
                args.experiment, args.dataset, args.subexp
            )
        )
    else:
        plt.savefig(
            "experiments/{}/results/{}/map_fig.png".format(
                args.experiment, args.dataset
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset", help="dataset name")
    parser.add_argument("--period", type=int, default=200, help="checkpoint period")
    parser.add_argument(
        "--experiment", type=str, default="maskrcnn", help="experiment name"
    )
    parser.add_argument("--subexp", type=str, help="sub experiment")
    args = parser.parse_args()

    generate_map_figure(args)
