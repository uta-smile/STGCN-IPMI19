"""Extracting video features from ."""
from __future__ import absolute_import, division, print_function

import glob
import os
import pathlib
import pickle
import random
import sys

import numpy as np
from smile import app, flags, logging

flags.DEFINE_string("dense_feature_path",
                    "/mnt/data/m2cai/m2cai_tool/dense_features/train",
                    "Path to dense features.")
flags.DEFINE_string("label_path", "/mnt/data/m2cai/m2cai_tool/train_dataset",
                    "Path to labels.")
flags.DEFINE_string("output_path", "video_features",
                    "Output path for video features.")
flags.DEFINE_string("output_name", "train_video.pkl",
                    "Output file name for video features.")
flags.DEFINE_integer("video_pad", 4, "Frames to pad to labelled frame.")
flags.DEFINE_float("valid_ratio", 0.1,
                   "Ratio of validation data from training data.")

FLAGS = flags.FLAGS


def get_video_features(feature_file, label_file, video_pad=4):
    """
    """
    # Get index list and label list from label file
    with open(label_file, "r") as f_reader:
        lines = f_reader.readlines()[1:]
    index = [int(x.split()[0]) for x in lines]
    labels = [np.fromstring(" ".join(list(x.split()[1:])), dtype=int, sep=" ") \
                for x in lines]
    # Get video features from index list
    feature_and_label = []
    frame_dense_features = pickle.load(open(feature_file, "rb"))
    total_num_frame, dense_feature_len = frame_dense_features.shape
    for i in range(len(index)):
        # Get current index
        frame_idx = index[i]
        frame_len = 2 * video_pad + 1
        # Empty feature ndarray
        video_feature = np.zeros((frame_len, dense_feature_len))
        start_idx = max(0, frame_idx - video_pad)
        end_idx = min(total_num_frame, frame_idx + video_pad + 1)
        actual_len = end_idx - start_idx
        # Copy dense features from video
        video_feature[0:actual_len, :] = \
            frame_dense_features[start_idx:end_idx, :]
        feature_and_label.append((video_feature, labels[i]))
    return feature_and_label


def get_data_and_label_file(feature_path, label_path):
    label_files = glob.glob(os.path.join(label_path, "*.txt"))
    label_files.sort()
    assert len(label_files) > 0, "No label files found."
    feature_files = []
    for label_file in label_files:
        data_label_pair = []
        file_index = label_file.split("/")[-1].split(".")[0].split("-")[0]
        feature_file = glob.glob(
            os.path.join(feature_path, "%s_dense.pkl" % file_index))
        assert len(feature_file) == 1, "Invalid feature files found."
        feature_files.extend(feature_file)
    return zip(feature_files, label_files)


def main(_):
    """
    """
    # Get all pickle files and related labels.
    assert os.path.isdir(FLAGS.dense_feature_path), "Invalid feature path."
    assert os.path.isdir(FLAGS.label_path), "Invalid label path."
    data_and_label_file = get_data_and_label_file(FLAGS.dense_feature_path,
                                                  FLAGS.label_path)
    # Get video feature and label for each pickle file (video).
    feature_and_label = []
    for feature_file, label_file in data_and_label_file:
        logging.info("Processing video %s" % \
                        label_file.split("/")[-1].split(".")[0])
        logging.info("Label file is %s" % label_file)
        logging.info("Feature file is %s" % feature_file)
        data = get_video_features(
            feature_file, label_file, video_pad=FLAGS.video_pad)
        feature_and_label.extend(data)
        logging.info("Processing finished, %d samples collected" % \
                        len(feature_and_label))
    # Divide training and validation if needed.
    # The ugliest way: shuffle then divide.
    if FLAGS.valid_ratio > 0:
        logging.info("Split %f as validation set." % FLAGS.valid_ratio)
        random.shuffle(feature_and_label)
    valid_set = feature_and_label[:int(FLAGS.valid_ratio * \
                                    len(feature_and_label))]
    train_set = feature_and_label[int(FLAGS.valid_ratio * \
                                    len(feature_and_label)):]
    # Save to output path.
    if not os.path.isdir(FLAGS.output_path):
        pathlib.Path(FLAGS.output_path).mkdir(parents=True, exist_ok=True)
    if len(valid_set) > 0:
        pickle.dump(
            valid_set,
            open(os.path.join(FLAGS.output_path, "valid_video.pkl"), "wb"))
    pickle.dump(train_set,
                open(os.path.join(FLAGS.output_path, FLAGS.output_name), "wb"))


if __name__ == "__main__":
    app.run()
