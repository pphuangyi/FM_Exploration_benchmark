"""
Evaluation metrics
"""

import torch

def compute_roc(scores,
                labels,
                num_thresholds = 100,
                vmin           = None,
                vmax           = None,
                reverse        = False):
    """
    Compute the Receiver Operating Characteristic (ROC)
    and area under the ROC curve (AUC) with a given number of thresholds
    """
    if len(scores) != len(labels):
        raise ValueError("Scores and labels must be of the same length.")

    scores = scores.clone().detach()
    labels = labels.clone().detach()

    if vmin is None:
        vmin = scores.min()
    if vmax is None:
        vmax = scores.max()

    thresholds = torch.linspace(vmin, vmax, num_thresholds)
    tpr_list = []  # True Positive Rate
    fpr_list = []  # False Positive Rate

    for thresh in thresholds:

        if reverse:
            predicted = (scores < thresh).int()
        else:
            predicted = (scores > thresh).int()

        TP = ((predicted == 1) & (labels == 1)).sum().item()
        FP = ((predicted == 1) & (labels == 0)).sum().item()
        FN = ((predicted == 0) & (labels == 1)).sum().item()
        TN = ((predicted == 0) & (labels == 0)).sum().item()

        TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) != 0 else 0

        tpr_list.append(TPR)
        fpr_list.append(FPR)

    # Convert to tensors for sorting and AUC calculation
    fpr_tensor = torch.tensor(fpr_list)
    tpr_tensor = torch.tensor(tpr_list)
    sorted_indices = torch.argsort(fpr_tensor)

    sorted_fpr = fpr_tensor[sorted_indices]
    sorted_tpr = tpr_tensor[sorted_indices]

    auc = torch.trapz(sorted_tpr, sorted_fpr).item()

    return sorted_fpr.tolist(), sorted_tpr.tolist(), auc


def compute_pr(scores,
               labels,
               num_thresholds = 100,
               vmin           = None,
               vmax           = None,
               reverse        = False):
    """
    Compute the Precision-Recall Curve and Average precision.
    """
    if len(scores) != len(labels):
        raise ValueError("Scores and labels must be of the same length.")

    scores = scores.clone().detach()
    labels = labels.clone().detach()

    if vmin is None:
        vmin = scores.min()
    if vmax is None:
        vmax = scores.max()

    thresholds = torch.linspace(vmin, vmax, num_thresholds)
    precision_list = []
    recall_list = []

    for thresh in thresholds:
        if reverse:
            predicted = (scores < thresh).int()
        else:
            predicted = (scores > thresh).int()

        TP = ((predicted == 1) & (labels == 1)).sum().item()
        FP = ((predicted == 1) & (labels == 0)).sum().item()
        FN = ((predicted == 0) & (labels == 1)).sum().item()

        precision = TP / (TP + FP) if (TP + FP) != 0 else 1.0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0

        precision_list.append(precision)
        recall_list.append(recall)

    # Sort recall and precision for computing average precision
    recall_tensor = torch.tensor(recall_list)
    precision_tensor = torch.tensor(precision_list)
    sorted_indices = torch.argsort(recall_tensor)

    sorted_recall = recall_tensor[sorted_indices]
    sorted_precision = precision_tensor[sorted_indices]

    average_precision = torch.trapz(sorted_precision, sorted_recall).item()

    return sorted_recall.tolist(), sorted_precision.tolist(), average_precision
