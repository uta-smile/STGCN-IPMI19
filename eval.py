from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import numpy as np
import os
import pathlib

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from smile import app
from smile import flags
from smile import logging

from dataset import get_image_loader, get_segment_dense_loader
from metrics import m2cai_map
from models import get_image_model, get_video_model
from utils import get_gt_from_files, from_lines_to_list

flags.DEFINE_string("feature_type",
                    "video",
                    "Feature type: either image or video.")
# flags.DEFINE_string("feature_type",
#                     "image",
#                     "Feature type: either image or video.")
flags.DEFINE_string("model_path",
                    "saved_video_net/params_epoch_0200.pkl",
                    "Model path to load.")
# flags.DEFINE_string("model_path",
#                     "saved_video_net_l2/params_epoch_0200.pkl",
#                     "Model path to load.")
# flags.DEFINE_string("model_path",
#                     "saved_video_net_max/params_epoch_0200.pkl",
#                     "Model path to load.")
# flags.DEFINE_string("model_path",
#                     "saved_model_single_gpu/params_epoch_100.pkl",
#                     "Model path to load.")
flags.DEFINE_string("test_data",
                    "video_features/test_video.pkl",
                    "Test file.")
# flags.DEFINE_string("test_data",
#                     "/mnt/data/m2cai/m2cai_tool/images_again/dumped/test_00.pkl",
#                     "Test file.")
flags.DEFINE_integer("num_gpu", 1, "Number of gpus to use.")
flags.DEFINE_integer("num_test_videos", 5, "Number of testing videos.")
flags.DEFINE_integer("num_classes", 7, "Number of classes.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_string("output_path", "results_video", "The output path.")
flags.DEFINE_string("pool_type", "avg", "Pooling type for video net.")
# flags.DEFINE_string("pool_type", "max", "Pooling type for video net.")
# flags.DEFINE_string("pool_type", "l2", "Pooling type for video net.")
flags.DEFINE_string("gt_path", "gt", "Ground truth path.")

FLAGS = flags.FLAGS

MAGIC_NUMS = [4410, 2033, 2399, 1874, 1825]

HEADER_LINE = "Frame Grasper Bipolar Hook Scissors Clipper Irrigator SpecimenBag\n"
RESULT_FILE = "tool_video_{}_result.txt"

def final_evaluate(model, data_loader, data_len, num_classes, batch_size):
    logging.info("Getting final prediction.")
    model.eval()

    final_pred = np.zeros((data_len, num_classes))
    final_pred = torch.Tensor(final_pred)
    final_pred = final_pred.cuda()

    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.cuda()
        data = Variable(data)
        output = model(data)
        final_pred[batch_idx*batch_size:(batch_idx+1)*batch_size] = output.data
    logging.info("Final prediction ready.")
    return final_pred.cpu().numpy()

def write_as_m2cai_format(final_pred):
    """Write as m2cai format.
    """
    # Check the file path.
    if not os.path.isdir(FLAGS.output_path):
        pathlib.Path(FLAGS.output_path).mkdir(parents=True, exist_ok=True)
    # Accumulate the Magic List
    num_sum = 0
    sum_list = [0]
    for magic_num in MAGIC_NUMS:
        num_sum += magic_num
        sum_list.append(num_sum)
    # Write to files
    for video_idx in range(FLAGS.num_test_videos):
        logging.info("Writing video %d." % video_idx)
        output_lines = HEADER_LINE
        frame_idx = 0
        for idx in range(sum_list[video_idx], sum_list[video_idx+1]):
            output_lines += "%d %s\n" % (frame_idx,
                                         " ".join(map(str, final_pred[idx, :])))
            frame_idx += 25
        with open(os.path.join(
                    FLAGS.output_path, RESULT_FILE.format(video_idx + 11)),
                  "w") as f_writer:
            f_writer.writelines(output_lines)

def evaluate_from_files():
    gt_path = "gt"
    pred_path = "results_video"
    gt_files = ["tool_video_%d.txt" % x for x in range(11, 16)]
    pred_files = ["tool_video_%d_result.txt" % x for x in range(11, 16)]
    gt_files = [os.path.join(gt_path, gt_file) for gt_file in gt_files]
    pred_files = [os.path.join(pred_path, pred_file) \
                    for pred_file in pred_files]

    gt, pred = [], []
    # Get pred & gt from files
    for i in range(len(gt_files)):
        with open(gt_files[i], "r") as gt_reader, \
             open(pred_files[i], "r") as pred_reader:
            gt_lines = gt_reader.readlines()
            pred_lines = pred_reader.readlines()
            gt.extend(from_lines_to_list(gt_lines, dtype=int))
            pred.extend(from_lines_to_list(pred_lines, sep=" ", dtype=float))
    # Get mAP
    ap = m2cai_map(np.array(pred), np.array(gt))
    logging.info(ap)
    logging.info(sum(ap) / len(ap))

def get_gt_from_files(gt_path):
    gt_files = glob.glob(os.path.join(gt_path, "*.txt"))
    gt_files.sort()
    
    gt = []
    for gt_file in gt_files:
        with open(gt_file, "r") as gt_reader:
            gt_lines = gt_reader.readlines()
            gt.extend(from_lines_to_list(gt_lines, dtype=int))
    return np.array(gt)

def main(_):
    """Main function for final results evaluation.
    """
    # Get model.
    if FLAGS.feature_type is "image":
        model = get_image_model(num_gpus=FLAGS.num_gpu,
                                load_model_path=FLAGS.model_path)
        test_loader, test_data_len = get_image_loader(
                                        FLAGS.test_data,
                                        batch_size=FLAGS.batch_size,
                                        shuffle=False) # shuffle should never be true ??? Not really
    else:
        model = get_video_model(num_gpus=FLAGS.num_gpu,
                                load_model_path=FLAGS.model_path,
                                pool_type=FLAGS.pool_type)
        test_loader, test_data_len = get_segment_dense_loader(
                                        FLAGS.test_data,
                                        batch_size=FLAGS.batch_size,
                                        shuffle=False)
    # Evaluation.
    final_pred = final_evaluate(model, test_loader, test_data_len,
                                num_classes=FLAGS.num_classes,
                                batch_size=FLAGS.batch_size)

    gt = get_gt_from_files(FLAGS.gt_path)
    ap = m2cai_map(final_pred, gt)
    logging.info(ap)
    logging.info(sum(ap) / len(ap))
    # Write into files.
    # write_as_m2cai_format(final_pred)
    # # Get final results.
    # evaluate_from_files()

if __name__ == "__main__":
    app.run()
