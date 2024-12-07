import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import imageio
from einops import repeat
from icecream import ic
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calculate focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return: loss
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)

def l2_regularisation(m):
    l2_reg = None
    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def save_mask_prediction_example(mask, pred, iter):
    plt.imshow(pred[0, :, :], cmap='Greys')
    plt.savefig(f'images/{iter}_prediction.png')
    plt.imshow(mask[0, :, :], cmap='Greys')
    plt.savefig(f'images/{iter}_mask.png')

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0, 0, 0, 0), lw=2))

def iou_score_cal(output, target):
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
    dice = (2 * iou) / (iou + 1)
    return iou

def mask_IoU(prediction, groundtruth):
    prediction = prediction.detach().cpu().numpy()
    groundtruth = groundtruth.detach().cpu().numpy()
    intersection = np.logical_and(groundtruth, prediction)
    union = np.logical_or(groundtruth, prediction)
    if np.sum(union) == 0:
        return 1
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def generalized_energy_distance_iou(predictions, masks):
    n = predictions.shape[0]
    m = masks.shape[0]
    d1 = d2 = d3 = 0
    for i in range(n):
        for j in range(m):
            d1 += (1 - mask_IoU(predictions[i], masks[j]))

    for i in range(n):
        for j in range(n):
            d2 += (1 - mask_IoU(predictions[i], predictions[j]))

    for i in range(m):
        for j in range(m):
            d3 += (1 - mask_IoU(masks[i], masks[j]))

    d1 *= (2 / (n * m))
    d2 *= (1 / (n * n))
    d3 *= (1 / (m * m))

    ed = d1 - d2 - d3
    scores = mask_IoU(predictions[0], masks[0])

    return ed, scores

def dice_score_cal(pred, targs):
    pred = (pred > 0).float()
    intersection = (pred * targs).sum()
    union = pred.sum() + targs.sum()
    if union == 0:
        return 1.0  # If both prediction and target are zero, return 1
    dice_score = 2. * intersection / union
    return dice_score

def dice_coef_cal(output, target):
    smooth = 1e-5
    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

def iou(pred, true):
    """Calculate IOU, input as PyTorch tensors"""
    pred_bool = pred.bool().detach().cpu()
    true_bool = true.bool().detach().cpu()
    intersection = (pred_bool & true_bool).float().sum()
    union = (pred_bool | true_bool).float().sum()
    if union == 0 and intersection == 0:
        return 1
    else:
        return intersection / union

def hm_iou_cal(preds, trues):
    """Calculate Hungarian-Matched IOU, input as lists of PyTorch tensors"""
    num_preds = len(preds)
    num_trues = len(trues)
    cost_matrix = torch.zeros((num_preds, num_trues))
    for i, pred in enumerate(preds):
        for j, true in enumerate(trues):
            cost_matrix[i, j] = 1 - iou(pred, true)
    row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())
    matched_iou = [iou(preds[i], trues[j]) for i, j in zip(row_ind, col_ind)]
    avg_iou = torch.tensor(matched_iou).mean().item()
    return avg_iou, matched_iou

def calculate_dice_loss(inputs, targets, num_masks=5):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

def calculate_sigmoid_focal_loss(inputs, targets, num_masks=5, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks

def mean_dice_cal(pred_list, label_four):
    n = len(pred_list)
    m = len(label_four)
    dice = 0
    for i in range(n):
        for j in range(m):
            dice += dice_score_cal(pred_list[i].to(dtype=torch.float).squeeze().cpu().detach(), label_four[j].squeeze(0).cpu().detach())
    dice_mean = dice / (n * m)
    return dice_mean

def dice_max_cal1(pred_eval, label_four):
    dice_max = 0
    for i in range(pred_eval.shape[0]):
        dice_max_iter = 0
        for j in range(label_four.shape[0]):
            dice_score_iter = dice_score_cal(pred_eval[i].to(dtype=torch.float).squeeze().cpu().detach(), label_four[j].squeeze(0).cpu().detach())
            if j == 0:
                dice_max_iter = dice_score_iter
            else:
                if dice_score_iter > dice_max_iter:
                    dice_max_iter = dice_score_iter
        dice_max += dice_max_iter
    return dice_max / pred_eval.shape[0]

def dice_max_cal2(pred_eval, label_four):
    dice_max = -1
    for i in range(pred_eval.shape[0]):
        for j in range(label_four.shape[0]):
            dice_score_iter = dice_score_cal(pred_eval[i].to(dtype=torch.float).squeeze().cpu().detach(), label_four[j].squeeze(0).cpu().detach())
            if dice_score_iter > dice_max:
                dice_max = dice_score_iter
    return dice_max

def dice_avg_cal(pred_list, label_four):
    dice_all = 0
    pred_stack = torch.stack(pred_list, dim=0)
    pred_avg = (pred_stack > 0).cpu().detach()
    pred_avg = torch.where(pred_avg, torch.tensor(1), torch.tensor(0))
    pred_avg = torch.mean(pred_stack, dim=0)
    pred_avg = (pred_avg > 0).cpu().detach()
    pred_avg = torch.where(pred_avg, torch.tensor(1), torch.tensor(0))
    for i in range(label_four.shape[0]):
        dice_score_iter = dice_score_cal(pred_avg.to(dtype=torch.float).squeeze().cpu().detach(), label_four[i].squeeze(0).cpu().detach())
        dice_all += dice_score_iter
    return dice_all / label_four.shape[0], pred_avg
