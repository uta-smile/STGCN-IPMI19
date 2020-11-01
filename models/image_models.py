"""Models for image classification."""
import os

import torch
import torch.nn as nn
import torchvision

from models.densenet import densenet121
import pretrainedmodels as pm

from smile import logging

class DenseNet121(nn.Module):
    """DenseNet121, it could also output dense features if needed.
    """
    def __init__(self, num_classes=7, with_dense_features=False):
        super(DenseNet121, self).__init__()
        if not with_dense_features:
            self.densenet = torchvision.models.densenet121(pretrained=True)
        else:
            self.densenet = densenet121(pretrained=True)
        self.densenet.classifier = nn.Linear(
                                    self.densenet.classifier.in_features,
                                    num_classes)

    def forward(self, x):
        x = self.densenet(x)
        return x

class NasNet(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard NasNet
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, num_classes):
        super(NasNet, self).__init__()
        self.nasnet = pm.__dict__['nasnetalarge'](num_classes=1000,
                                                  pretrained='imagenet')
        dense_feature_num = self.nasnet.last_linear.in_features
        self.nasnet.last_linear = nn.Linear(dense_feature_num, num_classes)

    def forward(self, x):
        x = self.nasnet(x)
        return x


def get_image_model(model_name="densenet121", num_gpus=1, num_classes=7,
                    with_dense_features=False, load_model_path=None):
    """Get model.
    """
    if model_name == "densenet121":
        model = DenseNet121(with_dense_features=with_dense_features).cuda()
    elif model_name == "nasnet":
        model = NasNet(num_classes=num_classes).cuda()
    else:
        model = None
    logging.info(model)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    if os.path.isfile(load_model_path):
        logging.info("Loading model from %s" % load_model_path)
        model.load_state_dict(torch.load(load_model_path))
    return model
