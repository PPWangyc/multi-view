# This file contains the implementation of the r2 score metric
import logging
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import gammaln
from sklearn.metrics import r2_score as r2_score_sklearn
from torcheval.metrics import R2Score

logger = logging.getLogger(__name__)

r2_metric = R2Score()
def r2_score(y_true, y_pred, device="cpu"):
    r2_metric.reset()
    r2_metric.to(device)
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    r2_metric.update(y_pred, y_true)
    return r2_metric.compute().item()

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

def clip_contrastive_loss(similarity_matrix):
    """
    Compute CLIP's contrastive loss given a similarity matrix.
    The matrix contains cosine similarities of two sets of features.
    """
    labels = torch.arange(len(similarity_matrix)).to(similarity_matrix.device)
    forw_correct  = topk(similarity_matrix, labels, k=1)
    back_correct  = topk(similarity_matrix.t(), labels, k=1)
    loss_i = F.cross_entropy(similarity_matrix, labels)
    loss_t = F.cross_entropy(similarity_matrix.t(), labels)
    return {
        "contrast_loss": (loss_i + loss_t) / 2,
        "percent_correct": (forw_correct + back_correct) / 2,
        "forw_correct": forw_correct,
        "back_correct": back_correct,
        "loss_f": loss_i,
        "loss_b": loss_t
    }

def neg_log_likelihood(rates, spikes, zero_warning=True, threshold=1e-9):
    """Calculates Poisson negative log likelihood given rates and spikes.
    formula: -log(e^(-r) / n! * r^n)
           = r - n*log(r) + log(n!)

    Parameters
    ----------
    rates : np.ndarray
        numpy array containing rate predictions
    spikes : np.ndarray
        numpy array containing true spike counts
    zero_warning : bool, optional
        Whether to print out warning about 0 rate
        predictions or not

    Returns
    -------
    float
        Total negative log-likelihood of the data
    """
    assert (
            spikes.shape == rates.shape
    ), f"neg_log_likelihood: Rates and spikes should be of the same shape. spikes: {spikes.shape}, rates: {rates.shape}"

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]

    assert not np.any(np.isnan(rates)), "neg_log_likelihood: NaN rate predictions found"

    assert np.all(rates >= 0), "neg_log_likelihood: Negative rate predictions found"
    if np.any(rates == 0):
        if zero_warning:
            logger.warning(
                "neg_log_likelihood: Zero rate predictions found. Replacing zeros with 1e-9"
            )
        rates[rates == 0] = threshold
    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    return np.sum(result)

def bits_per_spike(rates, spikes, threshold=1e-9):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts

    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = neg_log_likelihood(rates, spikes, threshold=threshold)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = neg_log_likelihood(null_rates, spikes, threshold=threshold)
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)

def batchwise_cosine_similarity(Z,B):
    # https://www.h4pz.co/blog/2021/4/2/batch-cosine-similarity-in-pytorch-or-numpy-jax-cupy-etc
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def batch_wise_sim_matrix(z):
    sim_matrix = z @ z.T
    return sim_matrix

def batch_wise_contrastive_loss(sim_matrix):
    N = sim_matrix.shape[0]
    # remove the diagonal from the sim_matrix
    mask = torch.eye(N, dtype=torch.bool, device=sim_matrix.device)
    sim_matrix = sim_matrix[~mask].view(N, N-1)
    labels = torch.arange(N).to(sim_matrix.device)
    labels_i, labels_j = labels[:N//2], labels[N//2:] -1
    labels = torch.cat([labels_j, labels_i]).to(sim_matrix.device)
    loss = F.cross_entropy(sim_matrix, labels)
    percent_correct = topk(sim_matrix, labels, k=1)
    return{
        "infoNCE_loss": loss,
        "percent_correct": percent_correct
    }

def byol_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def variance_loss_fn(x, y):
    # NOTE: x, and y are already centered
    # variance across batch samples
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    var_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
    return var_loss

def covariance_loss_fn(x, y):
    # NOTE: x, and y are already centered
    N = x.shape[0]
    D = x.shape[-1]
    cov_x = (x.T @ x) / (N - 1)
    cov_y = (y.T @ y) / (N - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
        D
    ) + off_diagonal(cov_y).pow_(2).sum().div(D)
    return cov_loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def compute_varexp(y_true, y_pred):
    """variance explained of y_true by y_pred across axis=0"""
    y_var = ((y_true - y_true.mean(axis=0)) ** 2).mean(axis=0)
    residual = ((y_true - y_pred) ** 2).mean(axis=0)
    varexp = 1 - residual / y_var
    return varexp    

# metrics list, return different metrics results
def metrics_list(gt, pred, metrics=["bps", "r2", "rsquared", "mse", "mae", "acc"], device="cpu"):
    results = {}

    if "bps" in metrics:
        gt, pred = gt.transpose(-1,0).cpu().numpy(), pred.transpose(-1,0).cpu().numpy()
        bps_list = []
        for i in range(gt.shape[-1]): 
            bps = bits_per_spike(pred[:,:,[i]], gt[:,:,[i]])
            if np.isinf(bps):
                bps = np.nan
            bps_list.append(bps)
        mean_bps = np.nanmean(bps_list)
        results["bps"] = mean_bps
    
    if "r2" in metrics:
        r2_list = []
        for i in range(gt.shape[0]):
            r2s = [r2_score(y_true=gt[i].T[k], y_pred=pred[i].T[k], device=device) for k in range(len(gt[i].T))]
            r2_list.append(np.ma.masked_invalid(r2s).mean())
        r2 = np.mean(r2_list)
        results["r2"] = r2
        
    if "rsquared" in metrics:
        r2 = 0
        for i in range(gt.shape[-1]):
            r2_list = []
            for j in range(gt.shape[0]):
                r2 = r2_score(y_true=gt[j,:,i], y_pred=pred[j,:,i], device=device) 
                r2_list.append(r2)
            r2 += np.mean(r2_list)
        results["rsquared"] = r2 / gt.shape[-1]
        
    return results

# get neuron metrics results
def compute_neuron_metrics(data_dict, metrics=["bps", "r2", "ve"], norm=True):
    results = {}
    gt, pred = data_dict["gt"], data_dict["pred"]
    
    # loop over the neurons
    if "bps" in metrics:
        num_neurons = gt.shape[-1]
        bps_list = []
        for i in range(num_neurons):
            bps = bits_per_spike(pred[:,:,[i]], gt[:,:,[i]])
            if np.isinf(bps):
                bps = np.nan
            bps_list.append(bps)
        bps_list = np.array(bps_list)
        results["bps"] = bps_list

    if "ve" in metrics:
        norm_gt, norm_pred = data_dict["norm_gt"], data_dict["norm_pred"] if "norm_gt" in data_dict else None
        # reshape the gt and pred
        if norm:
            _gt, _pred = norm_gt, norm_pred
        else:
            _gt, _pred = gt, pred
        _gt = _gt.reshape(-1, _gt.shape[-1])
        _pred = _pred.reshape(-1, _pred.shape[-1])
        ven = compute_varexp(y_true=_gt, y_pred=_pred)
        results["ve"] = ven

    if "r2" in metrics:
        if norm:
            _gt, _pred = norm_gt, norm_pred
        else:
            _gt, _pred = gt, pred
        _gt = _gt.reshape(-1, _gt.shape[-1])
        _pred = _pred.reshape(-1, _pred.shape[-1])
        r2 = r2_score_sklearn(y_true=_gt, y_pred=_pred, multioutput="raw_values")
        results["r2"] = r2
    return results


# ============================================================================
# Mask and Bounding Box Evaluation Metrics
# ============================================================================

def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks.
    
    Args:
        mask1: Binary mask (H, W) with dtype uint8
        mask2: Binary mask (H, W) with dtype uint8
    
    Returns:
        IoU score between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def compute_bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute IoU between two bounding boxes in [x, y, w, h] format.
    
    Args:
        bbox1: Bounding box [x, y, w, h]
        bbox2: Bounding box [x, y, w, h]
    
    Returns:
        IoU score between 0 and 1
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Convert to [x0, y0, x1, y1] format
    x1_min, y1_min = x1, y1
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_min, y2_min = x2, y2
    x2_max, y2_max = x2 + w2, y2 + h2
    
    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 1.0 if inter_area == 0 else 0.0
    
    return float(inter_area) / float(union_area)


def mask_to_bbox(mask: np.ndarray) -> np.ndarray:
    """Compute bounding box from binary mask using min/max of foreground pixels.
    
    Args:
        mask: Binary mask (H, W) with dtype uint8 (1 = foreground, 0 = background)
    
    Returns:
        Bounding box [x, y, w, h] in COCO format as numpy array
    """
    if mask.sum() == 0:
        # Empty mask, return zero bbox
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    # Find foreground pixel coordinates
    y_coords, x_coords = np.where(mask > 0)
    
    x_min = float(x_coords.min())
    y_min = float(y_coords.min())
    x_max = float(x_coords.max())
    y_max = float(y_coords.max())
    
    # Convert to [x, y, w, h] format
    return np.array([x_min, y_min, x_max - x_min + 1, y_max - y_min + 1])


def compute_precision_recall_f1(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> Dict[str, float]:
    """Compute precision, recall, and F1 score for binary masks.
    
    Args:
        pred_mask: Predicted binary mask (H, W)
        gt_mask: Ground truth binary mask (H, W)
    
    Returns:
        Dictionary with 'precision', 'recall', 'f1' scores
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    pred_positive = pred_mask.sum()
    gt_positive = gt_mask.sum()
    
    # Precision = TP / (TP + FP)
    if pred_positive == 0:
        precision = 1.0 if gt_positive == 0 else 0.0
    else:
        precision = float(intersection) / float(pred_positive)
    
    # Recall = TP / (TP + FN)
    if gt_positive == 0:
        recall = 1.0 if pred_positive == 0 else 0.0
    else:
        recall = float(intersection) / float(gt_positive)
    
    # F1 = 2 * (precision * recall) / (precision + recall)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }