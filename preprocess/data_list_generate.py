"""Script to generate dataset image lists (training, validation, testing lists)
from the extracted images using the results of video_process.py and dump them 
to pickle file if marked as needed."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import pathlib
import pickle
from PIL import Image
import numpy as np
import random
import sys

from smile import app
from smile import flags
from smile import logging


flags.DEFINE_string("train_image_path",
                    "/mnt/data/m2cai/m2cai_tool/images_299/train/",
                    "Data path of train images.")
flags.DEFINE_string("test_image_path",
                    "/mnt/data/m2cai/m2cai_tool/images_299/test/",
                    "Data path of test images.")
flags.DEFINE_string("target_path",
                    "/mnt/data/m2cai/m2cai_tool/images_299/image_list",
                    "Path to save the image lists.")
flags.DEFINE_float("valid_ratio", 0.1,
                   "Ratio of validation data from training data.")
flags.DEFINE_boolean("dump", True, "If to dump using pickle.")
flags.DEFINE_string("dump_target",
                    "/mnt/data/m2cai/m2cai_tool/images_299/dumped/",
                    "Target path for dumping.")
flags.DEFINE_integer("max_num_to_dump", 50000, "Max number to dump in a file.")
FLAGS = flags.FLAGS

def dump_image_list(image_list, target_path, max_num_to_dump=20000,
                    dataset_label="train"):
    logging.info("Processing %s set with %d images." \
                    % (dataset_label, len(image_list)))
    data_pairs = []
    idx = 0
    for i in range(len(image_list)):
        if i % 1000 == 0:
            logging.info("Processed %d images." % i)
        if i > 0 and i % max_num_to_dump == 0:
            logging.info("Dumping the %d file." % idx)
            pickle.dump(data_pairs,
                        open(os.path.join(target_path,
                                          "%s_%02d.pkl" % (dataset_label, idx)), 
                        "wb"))
            data_pairs = []
            idx += 1
        image_path = image_list[i].split("\t")[0]
        label_str = image_list[i].split("\t")[1]
        # image = np.asarray(Image.open(image_path))
        image = Image.open(image_path)
        label = np.fromstring(" ".join(list(label_str)),
                                  dtype=int, sep=" ")
        data_pairs += [(image, label)]
    if len(data_pairs) > 0:
        pickle.dump(data_pairs,
                    open(os.path.join(target_path,
                                      "%s_%02d.pkl" % (dataset_label, idx)), 
                        "wb"))

def main(_):
    train_list = []
    valid_list = []
    test_list = []

    undivided_train_images = glob.glob(os.path.join(FLAGS.train_image_path,
                                                    '*/*.jpg'))
    test_images = glob.glob(os.path.join(FLAGS.test_image_path, '*/*.jpg'))
    undivided_train_images.sort()
    test_images.sort()

    assert 0 < FLAGS.valid_ratio < 1, \
        "Validation ratio should be set as a number in (0, 1)."
    valid_images = random.sample(undivided_train_images, 
                                 int(FLAGS.valid_ratio * \
                                    len(undivided_train_images)))
    train_images = list(set(undivided_train_images) - set(valid_images))

    train_list = ['%s\t%s' % (x, x.split('/')[-1].split('.')[0].split('_')[-1])\
                    for x in train_images]
    valid_list = ['%s\t%s' % (x, x.split('/')[-1].split('.')[0].split('_')[-1])\
                    for x in valid_images]
    test_list = ['%s\t%s' % (x, x.split('/')[-1].split('.')[0].split('_')[-1])
                    for x in test_images]

    if not os.path.isdir(FLAGS.target_path):
        pathlib.Path(FLAGS.target_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(FLAGS.target_path, 'train.lst'), 'w') as f_writer:
        f_writer.write('\n'.join(train_list))
    with open(os.path.join(FLAGS.target_path, 'valid.lst'), 'w') as f_writer:
        f_writer.write('\n'.join(valid_list))
    with open(os.path.join(FLAGS.target_path, 'test.lst'), 'w') as f_writer:
        f_writer.write('\n'.join(test_list))
        
    if FLAGS.dump:
        if not os.path.isdir(FLAGS.dump_target):
            pathlib.Path(FLAGS.dump_target).mkdir(parents=True, exist_ok=True)
        # train
        dump_image_list(train_list, FLAGS.dump_target, dataset_label="train",
                        max_num_to_dump=FLAGS.max_num_to_dump)
        # valid
        dump_image_list(valid_list, FLAGS.dump_target, dataset_label="valid",
                        max_num_to_dump=FLAGS.max_num_to_dump)
        # test
        dump_image_list(test_list, FLAGS.dump_target, dataset_label="test",
                        max_num_to_dump=FLAGS.max_num_to_dump)

if __name__ == "__main__":
    app.run()
