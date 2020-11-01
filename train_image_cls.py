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
from torchvision import transforms

from dataset import get_image_loader
from metrics import m2cai_map
from models import get_image_model
from train_utils import evaluate, final_evaluate, train
from utils import get_gt_from_files

flags.DEFINE_integer("num_classes", 7, "Number of classes.")
flags.DEFINE_string("model_name", "densenet121", "Model name to use.")
flags.DEFINE_boolean("use_pretrained", True, "If used pretrained model.")
flags.DEFINE_boolean("train_list", "/mnt/data/m2cai/m2cai_tool/images_cropped/dumped/train*.pkl",
    "Train data set list.")
flags.DEFINE_string(
    "valid_list", "/mnt/data/m2cai/m2cai_tool/images_cropped/dumped/valid*.pkl",
    "Valid data set list.")
flags.DEFINE_string(
    "test_list", "/mnt/data/m2cai/m2cai_tool/images_cropped/dumped/test*.pkl",
    "Test data set list.")
flags.DEFINE_integer("num_gpu", 4, "Number of gpus to use.")
flags.DEFINE_integer("batch_size", 64, "Batch size.")
flags.DEFINE_float("lr", 0.0001, "Optimizer Learning Rate.")
flags.DEFINE_float("momentum", 0.5, "Optimizer momentum.")
flags.DEFINE_integer("epoch_num", 300, "Epoch numbers to train.")
flags.DEFINE_string("save_path", "saved/saved_image_model_with_adam",
                    "Path to save models.")
flags.DEFINE_string("load_model_path", "", "Save model parameters.")
flags.DEFINE_string("gt_path", "gt", "Groud truth path.")
flags.DEFINE_boolean("data_augment", True, "If to augment data.")
flags.DEFINE_string("loss_type", "bce",
                    "Loss type, BCE or MultiLabelSoftMarginLoss")

FLAGS = flags.FLAGS


def main(_):
    """Main function for image classification.
    """
    # Get model.
    logging.info("Creating model.")
    model = get_image_model(
        model_name=FLAGS.model_name,
        num_classes=FLAGS.num_classes,
        num_gpus=FLAGS.num_gpu,
        load_model_path=FLAGS.load_model_path)
    logging.info("Model is ready.")
    # Get data loaders.
    logging.info("Creating data loaders.")
    if FLAGS.data_augment:
        transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        transform = None
    train_loader, _ = get_image_loader(
        FLAGS.train_list,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        transform=transform)
    valid_loader, _ = get_image_loader(
        FLAGS.valid_list, batch_size=FLAGS.batch_size, shuffle=True)
    test_loader, test_data_len = get_image_loader(
        FLAGS.test_list, batch_size=FLAGS.batch_size, shuffle=False)
    logging.info("Data loaders are ready.")
    # Get opitimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    # Get criterion.
    if FLAGS.loss_type == "bce":
        criterion_train = nn.BCEWithLogitsLoss()
        criterion_test = nn.BCELoss()
        need_sigmoid = True
    else:
        criterion_train = criterion_test = nn.MultiLabelSoftMarginLoss()
        need_sigmoid = False
    # Scheduler.
    scheduler = ReduceLROnPlateau(optimizer, factor=0.7, patience=3, mode="min")
    # Start training.
    if not os.path.isdir(FLAGS.save_path):
        pathlib.Path(FLAGS.save_path).mkdir(parents=True, exist_ok=True)
    for i in range(FLAGS.epoch_num):
        train(i, model, train_loader, optimizer, criterion_train, scheduler)
        evaluate(model, valid_loader, criterion_test, need_sigmoid=need_sigmoid)
        if i % 10 == 0:
            path_to_save = os.path.join(FLAGS.save_path,
                                        "params_epoch_%04d.pkl" % i)
            torch.save(model.state_dict(), path_to_save)
    # Final evaluation after training finishes.
    pred = final_evaluate(model, test_loader, test_data_len, FLAGS.num_classes,
                          FLAGS.batch_size)
    gt = get_gt_from_files(FLAGS.gt_path)
    ap = m2cai_map(pred, gt)
    logging.info("Average precision:")
    logging.info(ap)
    logging.info(sum(ap) / len(ap))


if __name__ == "__main__":
    app.run()
