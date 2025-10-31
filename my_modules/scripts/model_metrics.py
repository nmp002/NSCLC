from collections import OrderedDict
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, average_precision_score, roc_curve, auc, \
    accuracy_score, balanced_accuracy_score, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score


def calculate_auc_roc(model, loader, print_results=False, make_plot=False):
    model.eval()
    outs = torch.tensor([])
    targets = torch.tensor([])
    with torch.no_grad():
        for x, target in loader:
            outs = torch.cat((outs, model(x).cpu().detach()), dim=0)
            targets = torch.cat((targets, target.cpu().detach()), dim=0)

            # Clean up for memory
            del x, target
            torch.cuda.empty_cache()

        thresholds, idx = torch.sort(outs.detach().squeeze())
        sorted_targets = targets[idx]
        tpr = []
        fpr = []
        acc = []
        for t in thresholds:
            positive_preds = thresholds >= t
            true_positives = torch.logical_and(positive_preds, sorted_targets == 1)
            true_negatives = torch.logical_and(~positive_preds, sorted_targets == 0)
            false_positives = torch.logical_and(positive_preds, sorted_targets == 0)
            false_negatives = torch.logical_and(~positive_preds, sorted_targets == 1)
            tpr.append((torch.sum(true_positives) / (torch.sum(sorted_targets == 1))).item())
            fpr.append((torch.sum(false_positives) / (torch.sum(sorted_targets == 0))).item())
            acc.append(((torch.sum(true_positives) + torch.sum(true_negatives)) / len(sorted_targets)).item())
        tpr.append(0.0)
        fpr.append(0.0)
        d_fpr = [f1 - f0 for f1, f0 in zip(fpr[:-1], fpr[1:])]
        a = [t * df for df, t in zip(d_fpr, tpr[:-1])]

    # Final metrics
    auc = sum(a)
    best_acc = max(acc)
    thresh = thresholds[acc.index(best_acc)]

    # Optional print of metrics
    if print_results:
        print(f'>>> AUC-ROC {auc:.2f} || '
              f'Best accuracy of {best_acc:.2f} at threshold of {thresh:.2f} <<<')

    # Optional figure creation
    if make_plot:
        # For plot
        fig, ax1 = plt.subplots()

        # ROC
        ax1.plot(fpr, tpr, 'r-', label='ROC')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_ylim([0, 1])
        ax1.tick_params(axis='both', labelcolor='r')

        # Accuracy
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1])
        ax2.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twiny()
        ax2.plot(thresholds, acc, 'b-', label='Accuracy')
        ax2.set_xlabel('Threshold')
        ax2.tick_params(axis='x', labelcolor='b')

        # Legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='lower right')

        return auc, best_acc, thresh, fig
    else:
        return auc, best_acc, thresh


def score_model(model, loader_or_tensors, loss_fn=None, print_results=False,
                make_plot=False, threshold_type='none', threshold=0.5):
    """
    Evaluate model. Supports threshold_type: 'roc', 'f1', 'both', 'none', 'fixed'
    If threshold_type == 'fixed', uses provided `threshold` for confusion/accuracy.
    loader_or_tensors may be a PyTorch loader or a tuple (outs_tensor, labels_tensor).
    Returns (scores_dict, fig) if make_plot True. fig is a single 3-panel figure (ROC, PR, Confusion).
    """
    from collections import OrderedDict
    import numpy as _np
    import matplotlib.pyplot as _plt
    from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay,
                                 average_precision_score, roc_curve, auc, accuracy_score,
                                 balanced_accuracy_score, confusion_matrix, RocCurveDisplay,
                                 ConfusionMatrixDisplay)

    def make_the_plots(preds, targets, fpr, tpr, outs):
        fig, (ax1, ax2, ax3) = _plt.subplots(1, 3, figsize=(20, 5))
        # ROC
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=scores['ROC-AUC']).plot(ax=ax1)
        ax1.set_title('ROC Curve')
        # Precision-Recall
        PrecisionRecallDisplay.from_predictions(targets, outs).plot(ax=ax2)
        ax2.set_title('Precision-Recall Curve')
        # Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(targets, preds).plot(ax=ax3)
        ax3.set_title('Confusion Matrix')
        return fig

    scores = OrderedDict()
    model.eval()

    # Gather outputs & targets
    if isinstance(loader_or_tensors, tuple):
        outs, targets = loader_or_tensors
        outs = outs.detach().cpu()
        targets = targets.detach().cpu()
    else:
        outs = torch.tensor([])
        targets = torch.tensor([])
        with torch.no_grad():
            for x, target in loader_or_tensors:
                if next(model.parameters()).is_cuda:
                    x, target = x.cuda(), target.cuda()
                out = model(x)
                outs = torch.cat((outs, out.cpu().detach()), dim=0)
                targets = torch.cat((targets, target.cpu().detach()), dim=0)

    # Loss (optional)
    if loss_fn is not None:
        try:
            scores['Loss'] = loss_fn(outs, targets).item()
        except Exception:
            scores['Loss'] = loss_fn(outs, targets.unsqueeze(1)).item()

    # ROC and PR metrics (threshold-independent)
    fpr, tpr, roc_thresholds = roc_curve(targets, outs, pos_label=1)
    scores['ROC-AUC'] = auc(fpr, tpr)

    precision, recall, pr_thresholds = precision_recall_curve(targets, outs)
    scores['Average Precision'] = average_precision_score(targets, outs)
    f1_vals = (2 * precision * recall) / (precision + recall + 1e-9)
    scores['F1 Score'] = float(_np.nanmax(f1_vals))
    scores['F1-Optimal Threshold'] = float(pr_thresholds[_np.argmax(f1_vals)]) if len(pr_thresholds) > 0 else 0.5
    scores['ROC-Optimal Threshold'] = float(roc_thresholds[_np.argmax(tpr - fpr)])

    # Determine threshold to use for classification/CM
    thr = None
    tt = threshold_type.lower() if isinstance(threshold_type, str) else threshold_type
    if tt == 'roc':
        thr = scores['ROC-Optimal Threshold']
    elif tt == 'f1':
        thr = scores['F1-Optimal Threshold']
    elif tt == 'fixed':
        thr = float(threshold)
    elif tt in ('both', 'none'):
        thr = None
    else:
        raise ValueError(f'Unrecognized threshold_type: {threshold_type}')

    if thr is not None:
        preds = torch.zeros_like(outs)
        preds[outs > thr] = 1
        scores['Threshold Used'] = float(thr)
        scores['Accuracy'] = accuracy_score(targets, preds)
        scores['Balanced Accuracy'] = balanced_accuracy_score(targets, preds)
        scores['Confusion Matrix'] = confusion_matrix(targets, preds)
        if make_plot:
            fig = make_the_plots(preds, targets, fpr, tpr, outs)
            return scores, fig
        return scores

    # both/none: return threshold-independent metrics and two sets if 'both'
    if tt == 'both':
        preds_roc = torch.zeros_like(outs)
        preds_roc[outs > scores['ROC-Optimal Threshold']] = 1
        preds_f1 = torch.zeros_like(outs)
        preds_f1[outs > scores['F1-Optimal Threshold']] = 1
        scores['ROC Threshold Accuracy'] = accuracy_score(targets, preds_roc)
        scores['F1 Threshold Accuracy'] = accuracy_score(targets, preds_f1)
        if make_plot:
            figs = {}
            figs['roc'] = make_the_plots(preds_roc, targets, fpr, tpr, outs)
            figs['f1'] = make_the_plots(preds_f1, targets, fpr, tpr, outs)
            return scores, figs
        return scores

    # default return
    return scores

