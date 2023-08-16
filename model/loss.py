import torch
import numpy as np

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[loss != loss] = 0   # delete Nan
    return loss.mean()


def masked_mape_loss(y_pred, y_true, dataset):
    if dataset == 'METR_LA' or dataset == 'PEMS-BAY':
        "Follow the mape loss in DCRNN"
        mask = (y_true != 0).float()
        mask /= mask.mean()
        loss = torch.abs(torch.div(y_true - y_pred, y_true))
        loss = loss * mask
        loss[loss != loss] = 0
        return loss.mean()
    else:
        "Follow the metrics in other papers for fair comparison"
        mask = torch.gt(y_true, 0)
        pred = torch.masked_select(y_pred, mask)
        true = torch.masked_select(y_true, mask)
        return torch.mean(torch.abs(torch.div((true - pred), true)))


def masked_rmse_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_true - y_pred, 2)
    loss = loss * mask
    loss[loss != loss] = 0
    return torch.sqrt(loss.mean())

def masked_mse_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_true - y_pred, 2)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()