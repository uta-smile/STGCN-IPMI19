"""Dataset Loaders including Image Dataset Loader, Single Video Dataset Loader
and Segment Dataset Loader."""
from __future__ import absolute_import, division, print_function

import os
import pathlib

import numpy as np
from torch.utils.data import DataLoader

from dataset import (M2CAIToolDataset, M2CAIToolSingleVideoDataset,
                     M2CAIToolVideoFeatureDataset,
                     M2CAIToolVideoFeatureGraphDataset)


def get_image_loader(image_list,
                     batch_size=16,
                     shuffle=False,
                     num_workers=0,
                     transform=None):
    """Get Image Data Loader.
    """
    if transform:
        data = M2CAIToolDataset(image_list, transform=transform)
    else:
        data = M2CAIToolDataset(image_list)
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, len(data)


def get_single_video_loader(video,
                            resize_shape=None,
                            batch_size=16,
                            shuffle=False,
                            num_workers=0,
                            resize=None,
                            transform=None,
                            crop=True):
    """Get Single Video Data Loader.
    """
    if transform:
        data = M2CAIToolSingleVideoDataset(
            video, resize_shape=resize, transform=transform, crop=crop)
    else:
        data = M2CAIToolSingleVideoDataset(
            video, resize_shape=resize, crop=crop)
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, len(data)


def get_segment_dense_loader(dense_file,
                             batch_size=16,
                             shuffle=False,
                             num_workers=0,
                             transform=None):
    """Get Segment Dense Data Loader.
    """
    if transform:
        data = M2CAIToolVideoFeatureDataset(dense_file, transform)
    else:
        data = M2CAIToolVideoFeatureDataset(dense_file)
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, len(data)


def get_dense_graph_loader(dense_file,
                           batch_size=16,
                           shuffle=False,
                           num_workers=0,
                           transform=None):
    """Get dense graph feature Data Loader.
    """
    data = M2CAIToolVideoFeatureGraphDataset(dense_file, transform)
    data_loader = DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader, len(data)
