from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from smile import app
from smile import logging

def m2cai_map(pred, gt, num_intervals=2000):
    assert pred.shape == gt.shape, \
        "The shapes of pred and gt should be the same."
    num_classes = gt.shape[1]
    ap = [0.0] * num_classes
    # Pred matrix re-scaling.
    pred /= np.max(np.abs(pred), axis = 0)

    for class_idx in range(num_classes):
        # Get class-wise results (pred and gt).
        class_gt = gt[:, class_idx]
        class_pred = pred[:, class_idx]
        # Compute the step.
        max_pred = np.max(class_pred)
        min_pred = np.min(class_pred)
        buf = (class_pred[:, np.newaxis] > \
                np.linspace(min_pred, max_pred, num_intervals)[np.newaxis, :])
        tp = np.sum((buf == class_gt[:, np.newaxis]) & (buf == 1), axis=0)
        fp = np.sum((buf != class_gt[:, np.newaxis]) & (buf == 1), axis=0)
        cls_num = np.sum(class_gt > 0)
        rec = tp / cls_num
        prec = tp / (tp + fp)
        notnan_idx = np.where(~np.isnan(prec))
        prec = prec[notnan_idx]
        rec = rec[notnan_idx]

        for t in np.arange(0, 1.01, 0.1):
            idx = (np.array(()), )
            threshold = 1e-5
            while len(idx[0]) == 0:
                idx = np.where(abs(rec - t) < threshold)
                threshold *= 2
            p = np.mean(prec[idx])
            ap[class_idx] = ap[class_idx] + p / 11
    return ap

def test_one():
    gt = np.array([[1, 0, 1, 0, 1],
                   [0, 1, 0, 1, 0],
                   [1, 0, 0, 1, 0]])
    pred = np.array([[-1.32, 0.65, 1.12, 0.02, -1.03],
                     [-1.01, -0.5, 0.0, 0.2, 0.04],
                     [1.23, 0.4, 0.5, 0.8, -0.4]])
    # ap = m2cai_map(pred, gt)
    ap = m2cai_map(pred, gt)
    logging.info(ap)
    logging.info(sum(ap) / len(ap))

def from_lines_to_list(lines, has_header=True, sep="\t", dtype=int):
    lines = lines[1:]
    results = []
    for line in lines:
        line_result = line.strip().split(sep)[1:]
        line_result = list(map(dtype, line_result))
        results.append(line_result)
    return results

def test_two():
    # Read gt and pred from files.
    gt_files = ["test_data/tool_video_01.txt",
                "test_data/tool_video_02.txt"]
    pred_files = ["test_data/tool_video_01_pred.txt",
                  "test_data/tool_video_02_pred.txt"]
    gt = []
    pred = []
    for i in range(len(gt_files)):
        with open(gt_files[i], "r") as gt_reader, \
             open(pred_files[i], "r") as pred_reader:
            gt_lines = gt_reader.readlines()
            pred_lines = pred_reader.readlines()
            gt.extend(from_lines_to_list(gt_lines, dtype=int))
            pred.extend(from_lines_to_list(pred_lines, dtype=float))
    # Call m2cai_map.
    ap = m2cai_map(np.array(pred), np.array(gt))
    logging.info(ap)
    logging.info(sum(ap) / len(ap))

def main(_):
    # logging.info("Test One:")
    test_one()
    # logging.info("Test Two:")
    test_two()

if __name__ == "__main__":
    app.run()
