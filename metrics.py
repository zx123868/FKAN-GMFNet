import numpy as np
import torch
import torch.nn.functional as F

'''
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    return iou, dice
'''


def iou_score(outputs, target):
    smooth = 1e-5

    def compute_iou_dice(output, target):
        # Ensure the inputs are tensors or numpy arrays
        if torch.is_tensor(output):
            output = torch.sigmoid(output).data.cpu().numpy()
        if torch.is_tensor(target):
            target = target.data.cpu().numpy()

        output_ = output > 0.5
        target_ = target > 0.5
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
        dice = (2 * iou) / (iou + 1)
        return iou, dice

    # Ensure pre_gt and outputs are processed correctly
    iou_outputs, dice_outputs = compute_iou_dice(outputs, target)

    return iou_outputs, dice_outputs


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
