"""Video Data Analysis according to annotation files."""
from __future__ import absolute_import, division, print_function

import glob
import os
import pickle

import cv2
import numpy as np
import skvideo.io
import torch
from PIL import Image
from sklearn.metrics import pairwise
from sklearn.preprocessing import normalize
from smile import logging
from torch.utils.data.dataset import Dataset
from torchvision import transforms

DEFAULT_TRANSFORM = transforms.Compose([transforms.ToTensor()])


def get_cropped_idx(img):
    """Given an image as np array format, crop the non-black-border part.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 32, 255, cv2.THRESH_BINARY)  # 32 is magic.
    idx = np.where(thresh == 255)
    x_start = np.min(idx[0])
    x_end = np.max(idx[0]) + 1
    y_start = np.min(idx[1])
    y_end = np.max(idx[1]) + 1
    return x_start, x_end, y_start, y_end


class M2CAIToolDataset(Dataset):
    """Dataset wrapping M2CAI tool detection images and multi-label groundtruth.
    """

    def __init__(self, image_pairs_files, transform=DEFAULT_TRANSFORM):
        """Init function for M2CAIToolDataset.

        Args:
            image_pairs_files: Image pairs dumped as files. It could be given as
                               either one dumped file or a glob search pattern.
            transform: If transform the image.
        """
        if os.path.isdir(image_pairs_files):
            logging.info("Loading dataset from dumped file %s" % \
                         image_pairs_file)
            self.image_pairs = pickle.load(open(image_pairs_files))
            logging.info("Loading finished.")
        else:
            dumped_files = glob.glob(image_pairs_files)
            assert len(dumped_files) > 0, "No dumped files found."
            self.image_pairs = []
            for dumped_file in dumped_files:
                logging.info("Loading dataset from dumped file %s" % \
                             dumped_file)
                self.image_pairs += pickle.load(open(dumped_file, "rb"))
                logging.info("Loading finished.")
        self.transform = transform

    def __getitem__(self, index):
        """Get item.
        """
        image = self.image_pairs[index][0]
        if self.transform is not None:
            image = self.transform(image)
        label = self.image_pairs[index][1].astype(np.float32)
        return image, label

    def __len__(self):
        """Len.
        """
        return len(self.image_pairs)


class M2CAIToolSingleVideoDataset(Dataset):
    """
    """

    def __init__(self,
                 video_path,
                 resize_shape=None,
                 transform=DEFAULT_TRANSFORM,
                 crop=True):
        """Init function for M2CAIToolSingleVideoDataset.
        """
        if crop:
            command_dict = {
                "-sws_flags": "bilinear",
                "-s": "%dx%d" % (360, 360)
            }  # Preprocess to 360x360
        else:
            command_dict = None
        self.transform = transform
        logging.info("Loading Video.")
        vid = skvideo.io.vread(video_path, outputdict=command_dict)
        self.vid = np.zeros(
            (vid.shape[0], resize_shape[0], resize_shape[1], 3), dtype=np.uint8)
        logging.info("Loading finished.")
        if crop:
            MAGIC_NUM = 1000
            x_0, x_1, y_0, y_1 = get_cropped_idx(vid[MAGIC_NUM, :, :, :])
            for i in range(self.vid.shape[0]):
                self.vid[i, :, :, :] = cv2.resize(vid[i, x_0:x_1, y_0:y_1, :],
                                                  resize_shape)

    def __getitem__(self, index):
        """Get item. Labels have no use in this case.
        """
        image = self.vid[index, :, :, :]
        if self.transform:
            image = self.transform(image)
        return image, 1

    def __len__(self):
        """Len.
        """
        return self.vid.shape[0]


class M2CAIToolVideoFeatureDataset(Dataset):
    """
    """

    def __init__(self, file_path, transform=None):
        """Init function for M2CAIToolSingleVideoDataset.
        """
        self.data = pickle.load(open(file_path, "rb"))
        self.transform = transform

    def __getitem__(self, index):
        data, label = self.data[index]
        if self.transform:
            data = self.transform(data)
        return data.astype(np.float32), label.astype(np.float32)

    def __len__(self):
        """Len.
        """
        return len(self.data)


class M2CAIToolVideoFeatureGraphDataset(Dataset):
    """
    """

    def __init__(self, file_path, transform=None):
        """Init function for M2CAIToolSingleVideoDataset.
        """
        self.data = pickle.load(open(file_path, "rb"))
        self.transform = transform

    def __getitem__(self, index):
        data, label = self.data[index]
        if self.transform:
            data = self.transform(data)
        adj = pairwise.cosine_similarity(data)
        # Normalize. After this, the adj matrix will not be symmetric.
        adj = normalize(adj)
        return data.astype(np.float32), adj.astype(np.float32), label.astype(
            np.float32)

    def __len__(self):
        """Len.
        """
        return len(self.data)
