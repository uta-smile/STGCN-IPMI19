"""Generate Video Data Statistics according to annotation files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import time
from collections import OrderedDict

from smile import app
from smile import flags
from smile import logging

flags.DEFINE_string("data_path",
                    "/mnt/data/m2cai/m2cai_tool/",
                    "Data path of M2CAI Tool Detection.")
flags.DEFINE_boolean("need_train_stats", True,
                     "If train statistics is needed.")
flags.DEFINE_boolean("need_test_stats", True,
                     "If test statistics is needed. The information should \
                     never be used for designing the training process or \
                     algorithm.")
flags.DEFINE_string("save_file_path", "stats.txt",
                    "Path of the data statistics file.")
flags.DEFINE_boolean("has_header", True,
                     "If the annotation file has a header line.")
flags.DEFINE_boolean("multi_count", False,
                     "Counting according to Multi-class or not.")
flags.DEFINE_string("data_name", "m2cai_tool", "Data set name.")

FLAGS = flags.FLAGS

INDEX_TO_LABEL = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper",
                  "Irrigator", "Specimen Bag"]
LABEL_DICT = {k:v for k, v in enumerate(INDEX_TO_LABEL)}


def data_stats():
    """Analysis of the tool detection dataset.
       Generate the statistics from the ground truth information."""
    output_file = FLAGS.save_file_path
    if os.path.isfile(output_file):
        current_time_label = "_".join(time.ctime().replace(":", "_").split())
        if output_file.endswith(".txt"):
            output_file = output_file.replace(
                            ".txt", "_" + current_time_label + ".txt")
        else:
            output_file = output_file + current_time_label + ".txt"

    results = ""

    if FLAGS.need_train_stats:
        results += "*********************Training**************************\n"
        current_path = os.path.join(FLAGS.data_path, "train_dataset")
        results += count_frames(current_path, stage_flag="training",
                                multi_count_mode=FLAGS.multi_count)
        results += "\n\n\n"
    if FLAGS.need_test_stats:
        results += "*********************Testing***************************\n"
        current_path = os.path.join(FLAGS.data_path, "test_dataset")
        results += count_frames(current_path, stage_flag="testing",
                                multi_count_mode=FLAGS.multi_count)
        results += "\n"
    if not FLAGS.need_train_stats and not FLAGS.need_test_stats:
        results += "*********************Dataset***************************\n"
        current_path = FLAGS.data_path
        results += count_frames(current_path, stage_flag="overall",
                                multi_count_mode=FLAGS.multi_count)

    logging.info(results)

    with open(output_file, "w") as f_writer:
        f_writer.write(results)

def count_frames(path_to_count, stage_flag, multi_count_mode=True):
    """Count frame/image information based on the annotation information.

    Args:
        path_to_count: string, path to the folder of annotation files.
                       It could be either training or testing.
        stage_flag: string, use for knowning training or testing data.
        multi_count_mode: bool, to identify counting strategy.

    Returns:
        results for printing.
    """
    results = ""
    total_stats_dict = {}
    total_num = 0
    annotation_files = glob.glob(os.path.join(path_to_count, "*.txt"))
    annotation_files.sort()
    for per_file in annotation_files:
        current_stats_dict = {}
        with open(per_file, "r") as f_reader:
            lines = f_reader.readlines()[1:]
        total_num += len(lines)
        for per_line in lines:
            if multi_count_mode:
                bin_index = "".join(per_line.strip().split()[1:])
                int_index = int(bin_index, 2)
                if int_index in current_stats_dict:
                    current_stats_dict[int_index] += 1
                else:
                    current_stats_dict[int_index] = 1
            else:
                all_index = per_line.strip().split()[1:]
                for index, value in enumerate(all_index):
                    if int(value):
                        if index in current_stats_dict:
                            current_stats_dict[index] += 1
                        else:
                            current_stats_dict[index] = 1
        results += "In %s video file %s: \
                   \n\tThe total number of frames with annotation is %d:\n" \
                   % (stage_flag, per_file.split("/")[-1], len(lines))
        current_ordered_dict = OrderedDict(sorted(current_stats_dict.items()))
        for key, value in current_ordered_dict.items():
            if key in total_stats_dict:
                total_stats_dict[key] += value
            else:
                total_stats_dict[key] = value
            results += "\tThe number of class index %d: %d\n" % (key, value)

    results += "The total number of %s frames with annotation is %d.\n" \
               % (stage_flag, total_num)
    results += "\n"
    results += "There are %d multi-label class in total.\n" \
               % len(total_stats_dict)
    total_ordered_dict = OrderedDict(sorted(total_stats_dict.items()))
    for key, value in total_ordered_dict.items():
        results += "\tThe number of class index %s: %d\n" % (bin(key), value)
    return results

def main(_):
    """Main function to call."""
    data_stats()

if __name__ == '__main__':
    app.run()
