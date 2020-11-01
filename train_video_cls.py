"""Extracting images according to annotations from given videos."""
from __future__ import absolute_import, division, print_function

import os
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from smile import app, flags, logging
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import get_segment_dense_loader
from metrics import m2cai_map
from models import get_video_model
from train_utils import evaluate, final_evaluate, train
from utils import get_gt_from_files

flags.DEFINE_integer("num_classes", 7, "Number of classes.")
flags.DEFINE_string(
    "train_list",
    "/mnt/data/m2cai/m2cai_tool/video_features_cropped_long/train_video.pkl",
    "Train data set list.")
flags.DEFINE_string(
    "valid_list",
    "/mnt/data/m2cai/m2cai_tool/video_features_cropped_long/valid_video.pkl",
    "Valid data set list.")
flags.DEFINE_string(
    "test_list",
    "/mnt/data/m2cai/m2cai_tool/video_features_cropped_long/test_video.pkl",
    "Test data set list.")
flags.DEFINE_integer("num_gpu", 1, "Number of gpus to use.")
flags.DEFINE_integer("batch_size", 64, "Batch size.")
flags.DEFINE_float("lr", 0.000001, "Optimizer Learning Rate.")
flags.DEFINE_float("momentum", 0.5, "Optimizer momentum.")
flags.DEFINE_integer("epoch_num", 200, "Epoch numbers to train.")
flags.DEFINE_string("save_model_path", "saved/saved_video_new_avg_11_contrast",
                    "Path to save models.")
flags.DEFINE_string("load_model_path", "", "Save model parameters.")
flags.DEFINE_string("pool_type", "avg", "Temporal pooling type.")
flags.DEFINE_string("gt_path", "gt", "Groud truth path.")
flags.DEFINE_boolean("moe", False, "If use MOE in the model.")
flags.DEFINE_string("loss_type", "bce", "Either BCE or MultiSoftMargin")
flags.DEFINE_integer("frame_num", 11, "Segment length.")

FLAGS = flags.FLAGS


def main(_):
    """Main function for video classification.
    """
    # Get model.
    logging.info("Creating model.")
    model = get_video_model(
        num_gpus=FLAGS.num_gpu,
        load_model_path=FLAGS.load_model_path,
        num_classes=FLAGS.num_classes,
        pool_type=FLAGS.pool_type,
        moe=FLAGS.moe,
        frame_num=FLAGS.frame_num)
    logging.info("Model is ready.")
    # Get data loaders.
    logging.info("Creating data loaders.")
    train_loader, _ = get_segment_dense_loader(
        FLAGS.train_list, batch_size=FLAGS.batch_size, shuffle=True)
    valid_loader, _ = get_segment_dense_loader(
        FLAGS.valid_list, batch_size=FLAGS.batch_size, shuffle=True)
    test_loader, test_data_len = get_segment_dense_loader(
        FLAGS.test_list, batch_size=FLAGS.batch_size, shuffle=False)
    logging.info("Data loaders are ready.")
    # Get optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    # Get criterion.
    if FLAGS.loss_type is "bce":
        criterion_train = nn.BCEWithLogitsLoss()
        criterion_val = nn.BCELoss()
        need_sigmoid = True
    else:
        criterion_train = criterion_val = nn.MultiLabelSoftMarginLoss()
        need_sigmoid = False
    # Scheduler.
    scheduler = ReduceLROnPlateau(optimizer, factor=0.9, patience=5, mode="min")
    # Start training.
    if not os.path.isdir(FLAGS.save_model_path):
        pathlib.Path(FLAGS.save_model_path).mkdir(parents=True, exist_ok=True)
    for i in range(FLAGS.epoch_num):
        train(
            i,
            model,
            train_loader,
            optimizer,
            criterion_train,
            scheduler=scheduler)
        evaluate(model, valid_loader, criterion_val, need_sigmoid=need_sigmoid)
        if i % 10 == 0:
            path_to_save = os.path.join(FLAGS.save_model_path,
                                        "params_epoch_%04d.pkl" % i)
            torch.save(model.state_dict(), path_to_save)
    # Final evaluation after training finishes.
    pred = final_evaluate(model, test_loader, test_data_len,
                          FLAGS.num_classes, FLAGS.batch_size)
    gt = get_gt_from_files(FLAGS.gt_path)
    ap = m2cai_map(pred, gt)
    logging.info("Average precision:")
    logging.info(ap)
    logging.info(sum(ap) / len(ap))


if __name__ == "__main__":
    app.run()
