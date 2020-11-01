"""GraphConvolution Layer from https://github.com/tkipf/pygcn.
"""
import math

import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # Using torch.matmul to support batch operation.
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


if __name__ == "__main__":
    # A toy test with no batch.
    graph_conv = GraphConvolution(4, 2)
    toy_input = torch.randn(5, 4)
    toy_adj = torch.randn(5, 5)
    toy_output = graph_conv(toy_input, toy_adj)
    # output check.
    assert toy_output.shape[0] == 5
    assert toy_output.shape[1] == 2
    print(toy_output)
    # A toy test with batch.
    toy_input = torch.randn(10, 5, 4)
    toy_adj = torch.randn(10, 5, 5)
    toy_output = graph_conv(toy_input, toy_adj)
    # output check.
    assert toy_output.shape[0] == 10
    assert toy_output.shape[1] == 5
    assert toy_output.shape[2] == 2
