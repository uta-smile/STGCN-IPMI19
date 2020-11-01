"""Utils Script.
"""
import cv2
import glob
import os
import numpy as np


def from_lines_to_list(lines, has_header=True, sep="\t", dtype=int):
    lines = lines[1:]
    results = []
    for line in lines:
        line_result = line.strip().split(sep)[1:]
        line_result = list(map(dtype, line_result))
        results.append(line_result)
    return results

def get_gt_from_files(gt_path):
    gt_files = glob.glob(os.path.join(gt_path, "*.txt"))
    gt_files.sort() # Super important
    
    gt = []
    for gt_file in gt_files:
        with open(gt_file, "r") as gt_reader:
            gt_lines = gt_reader.readlines()
            gt.extend(from_lines_to_list(gt_lines, dtype=int))
    return np.array(gt)

def get_cropped_ndarray(img):
    """Given an image as np array format, crop the non-black-border part.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 32, 255, cv2.THRESH_BINARY) # 32 is magic.
    idx = np.where(thresh == 255)
    x_start = np.min(idx[0])
    x_end = np.max(idx[0]) + 1
    y_start = np.min(idx[1])
    y_end = np.max(idx[1]) + 1
    crop = img[x_start:x_end, y_start:y_end]
    return crop
    
def get_cropped_idx(img):
    """Given an image as np array format, crop the non-black-border part.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 32, 255, cv2.THRESH_BINARY) # 32 is magic.
    idx = np.where(thresh == 255)
    x_start = np.min(idx[0])
    x_end = np.max(idx[0]) + 1
    y_start = np.min(idx[1])
    y_end = np.max(idx[1]) + 1
    return x_start, x_end, y_start, y_end
