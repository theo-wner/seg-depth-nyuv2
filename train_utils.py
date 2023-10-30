import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

"""
Defines utility functions and classes
"""

# LR = Initial_LR * (1 - iter / max_iter)^0.9
class PolyLR(_LRScheduler):
    '''
    Used for scheduling the learning rate
    '''
    def __init__(self, optimizer, max_iterations, power=0.9):
        self.current_iteration = 0
        self.max_iterations = max_iterations
        self.power = power
        super().__init__(optimizer)

    def get_lr(self):
        self.current_iteration += 1
        return [base_lr * (1 - self.current_iteration / self.max_iterations) ** self.power for base_lr in self.base_lrs]
    

class RMSLELoss(nn.Module):
    '''
    Used for calculating the RMSLE loss
    '''
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
            
    def forward(self, pred, actual, valid_mask):
        pred = pred[valid_mask]
        actual = actual[valid_mask]
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
    

'''
Used for calculating the depth metrics
'''
# Source: https://github.com/ashutosh1807/PixelFormer/blob/main/pixelformer/utils.py
def compute_depth_metrics(pred, actual):
    pred = pred.cpu().numpy()
    actual = actual.cpu().numpy()

    # Add small epsilon to pred to avoid division by 0
    pred += 1e-6

    rms = (actual - pred) ** 2
    rms = np.sqrt(rms.mean())

    abs_rel = np.mean(np.abs(actual - pred) / actual)

    err = np.abs(np.log10(pred) - np.log10(actual))
    log10 = np.mean(err)

    thresh = np.maximum((actual / pred), (pred / actual))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    # sq_rel = np.mean(((actual - pred) ** 2) / actual)
    # log_rms = (np.log(actual) - np.log(pred)) ** 2
    # log_rms = np.sqrt(log_rms.mean())
    # err = np.log(pred) - np.log(actual)
    # silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    return [rms, abs_rel, log10, d1, d2, d3]