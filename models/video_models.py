"""Models for video classification."""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from smile import logging
from torch.autograd import Variable

from models.layers import GraphConvolution


class VideoNet(nn.Module):
    """
    """

    def __init__(self,
                 frame_num=9,
                 dense_feature_num=1024,
                 hidden_num=2000,
                 num_classes=7,
                 dropout=True,
                 dropout_p=0.5,
                 pool_type="avg"):
        super(VideoNet, self).__init__()
        self.dense_feature_num = dense_feature_num
        if pool_type is "avg":
            self.pool = nn.AvgPool2d((frame_num, 1), stride=(1, 1))
        elif pool_type is "l2":
            self.pool = nn.LPPool2d(2, (frame_num, 1), stride=(1, 1))
        else:  # max pooling
            self.pool = nn.MaxPool2d((frame_num, 1), stride=(1, 1))
        self.fc1 = nn.Linear(dense_feature_num, hidden_num)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_num, num_classes)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.pool(x)
        x = x.view(-1, self.dense_feature_num)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class VideoNetMOE(nn.Module):
    """
    """

    def __init__(self,
                 frame_num=9,
                 dense_feature_num=1024,
                 hidden_num=2000,
                 num_classes=7,
                 dropout=True,
                 dropout_p=0.5,
                 pool_type="avg",
                 num_mixtures=3):
        super(VideoNetMOE, self).__init__()
        self.dense_feature_num = dense_feature_num
        if pool_type is "avg":
            self.pool = nn.AvgPool2d((frame_num, 1), stride=(1, 1))
        elif pool_type is "l2":
            self.pool = nn.LPPool2d(2, (frame_num, 1), stride=(1, 1))
        else:  # max pooling
            self.pool = nn.MaxPool2d((frame_num, 1), stride=(1, 1))
        # self.fc1 = nn.Linear(dense_feature_num, hidden_num)
        # self.dropout = nn.Dropout(p=dropout_p)
        # self.fc2 = nn.Linear(hidden_num, num_classes)
        self.num_mixtures = num_mixtures
        self.num_classes = num_classes
        self.gated_linear = nn.Linear(
            dense_feature_num, self.num_classes * (self.num_mixtures + 1))
        self.expert_linear = nn.Linear(dense_feature_num,
                                       self.num_classes * self.num_mixtures)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.pool(x)
        x = x.view(-1, self.dense_feature_num)
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        gated = self.gated_linear(x)
        gated = gated.view(-1, self.num_mixtures + 1)
        gated = F.softmax(gated, dim=1)
        expert = self.expert_linear(x)
        expert = expert.view(-1, self.num_mixtures)
        expert = F.sigmoid(expert)
        x = torch.sum(expert * gated[:, :self.num_mixtures], dim=1)
        x = x.view(-1, self.num_classes)
        # x = self.fc2(x)
        return x


class VideoGraphNet(nn.Module):
    """Video Net with Graph Convolution.
    """

    def __init__(self,
                 frame_num=11,
                 dense_feature_num=1024,
                 hidden_num=1024,
                 graph_feature_num=1024,
                 num_classes=7,
                 dropout=0.75):
        super(VideoGraphNet, self).__init__()

        self.frame_num = frame_num
        self.dense_feature_num = dense_feature_num
        self.hidden_num = hidden_num
        self.graph_feature_num = graph_feature_num
        self.dropout = dropout
        self.num_classes = num_classes
        # GCN layer 1.
        self.gc1 = GraphConvolution(self.dense_feature_num, self.hidden_num)
        # self.gc1 = GraphConvolution(self.dense_feature_num, graph_feature_num)
        # GCN layer 2.
        self.gc2 = GraphConvolution(self.hidden_num, self.graph_feature_num)
        # Pooling over the graph.
        self.pool = nn.AvgPool2d((self.frame_num, 1), stride=(1, 1))
        # TODO(shengwang): Try fully-connected layer after pooling.
        self.fc1 = nn.Linear(self.graph_feature_num, self.num_classes)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.pool(x)
        x = x.view(-1, self.graph_feature_num)
        x = self.fc1(x)
        return x


def get_video_model(num_gpus=1,
                    load_model_path=None,
                    frame_num=9,
                    dense_feature_num=1024,
                    hidden_num=2000,
                    num_classes=7,
                    dropout=True,
                    dropout_p=0.5,
                    pool_type="avg",
                    moe=False,
                    num_mixtures=5,
                    graph_model=False):
    if graph_model:
        logging.info("Using Graph Convolution Net.")
        model = VideoGraphNet().cuda()
    elif moe:
        logging.info("Using Video Net with MOE.")
        model = VideoNetMOE(
            pool_type=pool_type, num_mixtures=num_mixtures,
            frame_num=frame_num).cuda()
    else:
        logging.info("Using Video Net.")
        model = VideoNet(pool_type=pool_type, frame_num=frame_num).cuda()
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    if os.path.isfile(load_model_path):
        model.load_state_dict(torch.load(load_model_path))
    return model


if __name__ == "__main__":
    model = VideoGraphNet(
        frame_num=3, dense_feature_num=8, hidden_num=4, num_classes=2)
    data = torch.randn(5, 3, 8)
    adj = torch.randn(5, 3, 3)
    output = model(Variable(data), Variable(adj))
    print(output.shape)
