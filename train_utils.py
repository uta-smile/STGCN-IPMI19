"""Utils for model training."""
import numpy as np
import torch
import torch.nn.functional as F
from smile import logging
from torch.autograd import Variable


def train(epoch, model, data_loader, optimizer, criterion, scheduler=None):
    """
    """
    # Train status.
    logging.info("Training epoch %d" % epoch)
    model.train()

    losses = []
    logging.info("Epoch %d" % epoch)
    logging.info("Learning rate %f" % optimizer.param_groups[0]["lr"])
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx % 100 == 0:
            logging.info("Batch %d" % batch_idx)
        data = data.cuda()
        target = target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.data.mean())
    if scheduler:
        scheduler.step(np.mean(losses))
    logging.info("Epoch %d Loss: %.4f" % (epoch, np.mean(losses)))


def evaluate(model, data_loader, criterion, need_sigmoid=False):
    """
    """
    # Evaluate
    logging.info("Evaluating")
    model.eval()
    losses = []
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.cuda()
        target = target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        if need_sigmoid:
            output = F.sigmoid(output)
        loss = criterion(output, target)
        losses.append(loss.data.mean())
    logging.info("Testing set: Loss: %.4f\n" % np.mean(losses))


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
        final_pred[batch_idx * batch_size:(batch_idx + 1) *
                   batch_size] = output.data
    logging.info("Final prediction ready.")
    return final_pred.cpu().numpy()


""" TODO(shengwang): Please refactor this ugly code.
"""


def train_graph(epoch, model, data_loader, optimizer, criterion,
                scheduler=None):
    """
    """
    # Train status.
    logging.info("Training epoch %d" % epoch)
    model.train()

    losses = []
    logging.info("Epoch %d" % epoch)
    logging.info("Learning rate %f" % optimizer.param_groups[0]["lr"])
    for batch_idx, (data, adj, target) in enumerate(data_loader):
        if batch_idx % 100 == 0:
            logging.info("Batch %d" % batch_idx)
        data, adj, target = Variable(data.cuda()), Variable(
            adj.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data, adj)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.data.mean())
    if scheduler:
        scheduler.step(np.mean(losses))
    logging.info("Epoch %d Loss: %.4f" % (epoch, np.mean(losses)))


def evaluate_graph(model, data_loader, criterion, need_sigmoid=False):
    """
    """
    # Evaluate
    logging.info("Evaluating")
    model.eval()
    losses = []
    for batch_idx, (data, adj, target) in enumerate(data_loader):
        data, adj, target = Variable(data.cuda()), Variable(
            adj.cuda()), Variable(target.cuda())
        output = model(data, adj)
        if need_sigmoid:
            output = F.sigmoid(output)
        loss = criterion(output, target)
        losses.append(loss.data.mean())
    logging.info("Testing set: Loss: %.4f\n" % np.mean(losses))


def final_evaluate_graph(model, data_loader, data_len, num_classes, batch_size):
    logging.info("Getting final prediction.")
    model.eval()

    final_pred = np.zeros((data_len, num_classes))
    final_pred = torch.Tensor(final_pred)
    final_pred = final_pred.cuda()

    for batch_idx, (data, adj, _) in enumerate(data_loader):
        data, adj = Variable(data.cuda()), Variable(adj.cuda())
        output = model(data, adj)
        final_pred[batch_idx * batch_size:(batch_idx + 1) *
                   batch_size] = output.data
    logging.info("Final prediction ready.")
    return final_pred.cpu().numpy()
