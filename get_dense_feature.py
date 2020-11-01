"""Extracting images according to annotations from given videos."""
from __future__ import absolute_import, division, print_function

import glob
import os
import pathlib
import pickle
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from smile import app, flags, logging
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import get_single_video_loader
from models import get_image_model

flags.DEFINE_integer("num_classes", 7, "Number of classes.")
flags.DEFINE_string("model_name", "densenet121", "Model name to use.")
flags.DEFINE_string("video_path", "/mnt/data/m2cai/m2cai_tool/test_dataset",
                    "Video path.")
flags.DEFINE_boolean("use_pretrained", True, "If used pretrained model.")
flags.DEFINE_integer("num_gpu", 4, "Number of gpus to use.")
flags.DEFINE_integer("batch_size", 64, "Batch size.")
flags.DEFINE_string("output_path",
                    "/mnt/data/m2cai/m2cai_tool/dense_features/test",
                    "Path to save dense features for each video.")
flags.DEFINE_string("load_model_path",
                    "saved_model/params_epoch_110_cluster.pkl",
                    "Saved model parameters.")
flags.DEFINE_boolean("crop", True, "If to crop the non-black-board.")

FLAGS = flags.FLAGS


def get_dense_features(model,
                       data_loader,
                       data_len,
                       batch_size,
                       dense_feature_length=1024):
    """Get dense features.
    """
    logging.info("Getting dense feature.")
    model.eval()

    dense_features = np.zeros((data_len, dense_feature_length))
    logging.info("The shape of dense features is:")
    logging.info(dense_features.shape)
    dense_features = torch.Tensor(dense_features)
    dense_features = dense_features.cuda()
    for batch_idx, (data, _) in enumerate(data_loader):
        if batch_idx % 100 == 0:
            logging.info("Processing the %d batch." % batch_idx)
        data = data.cuda()
        data = Variable(data)
        outputs = model(
            data)  # outputs[0] is the feature and [1] is the output.
        dense_features[batch_idx*batch_size:(batch_idx+1)*batch_size,:] = \
            outputs[0].data
    return dense_features.cpu().numpy()


def get_video_list(video_path, video_ext="mp4"):
    """Get video list.
    """
    return glob.glob(os.path.join(video_path, "*" + video_ext))


def main(_):
    """
    """
    # Get the model, the model's forward() should contain the dense features.
    model = get_image_model(
        model_name=FLAGS.model_name,
        num_gpus=FLAGS.num_gpu,
        num_classes=FLAGS.num_classes,
        load_model_path=FLAGS.load_model_path,
        with_dense_features=True)
    # Get video list.
    video_files = get_video_list(FLAGS.video_path)
    assert len(video_files) > 0, "No video found!"
    video_files.sort()
    if not os.path.isdir(FLAGS.output_path):
        pathlib.Path(FLAGS.output_path).mkdir(parents=True, exist_ok=True)
    for video_file in video_files:
        logging.info("Getting dense features from video %s" % video_file)
        video_loader, data_len = get_single_video_loader(
            video_file,
            batch_size=FLAGS.batch_size,
            resize=(224, 224),
            crop=FLAGS.crop)
        dense_features = get_dense_features(model, video_loader, data_len,
                                            FLAGS.batch_size)
        output_file_name = video_file.split("/")[-1].split(".")[0]
        output_file_name += "_dense.pkl"
        output_file_name = os.path.join(FLAGS.output_path, output_file_name)
        pickle.dump(dense_features, open(output_file_name, "wb"))


if __name__ == "__main__":
    app.run()
