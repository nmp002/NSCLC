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


def score_model(model, loader, loss_fn=None, print_results=False, make_plot=False, threshold_type='none'):
    """
    score_model now accepts either:
      - a DataLoader yielding (x, y) batches (original behavior), or
      - a tuple (scores_tensor, labels_tensor) where scores_tensor is 1D or Nx1 tensor
        of sigmoid-style outputs and labels_tensor is 1D tensor of 0/1 labels.

    Returns the same `scores` dict and `fig_out` (if make_plot True) as before.
    """
    def make_the_plots():
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=scores['ROC-AUC']).plot(ax=ax1)
        ax1.set_title('ROC')
        PrecisionRecallDisplay.from_predictions(targets, outs).plot(ax=ax2)
        ax2.set_title('Precision Recall Curve')
        ConfusionMatrixDisplay.from_predictions(targets, preds).plot(ax=ax3)
        ax3.set_title('Confusion Matrix')
        return fig

    scores = OrderedDict()

    # Detect raw (scores, labels) input for patient-wise use
    if isinstance(loader, tuple) and len(loader) == 2:
        # Accept either torch.Tensor or array-like inputs
        outs = loader[0].detach().cpu().squeeze() if isinstance(loader[0], torch.Tensor) else torch.tensor(loader[0])
        targets = loader[1].detach().cpu().squeeze() if isinstance(loader[1], torch.Tensor) else torch.tensor(loader[1])

        # Ensure float tensors
        outs = outs.float()
        targets = targets.long()
        loss = 0
        # If loss_fn provided and compatible, try to compute; otherwise skip
        if loss_fn is not None:
            try:
                loss = loss_fn(outs, targets)
                scores['Loss'] = loss
            except Exception:
                try:
                    scores['Loss'] = loss_fn(outs, targets.unsqueeze(1))
                except Exception:
                    # incompatible loss with raw scores, ignore
                    pass
    else:
        # Original behavior: iterate through loader and run model
        model.eval()
        outs = torch.tensor([])
        targets = torch.tensor([])
        loss = 0
        with torch.no_grad():
            for x, target in loader:
                if next(model.parameters()).is_cuda:
                    x, target = x.cuda(), target.cuda()
                outs = torch.cat((outs, model(x).cpu().detach()), dim=0)
                targets = torch.cat((targets, target.cpu().detach()), dim=0)

                # Clean up for memory
                del x, target
                torch.cuda.empty_cache()

            if loss_fn is not None:
                try:
                    loss += loss_fn(outs, targets)
                except Exception as e:
                    loss += loss_fn(outs, targets.unsqueeze(1))
                scores['Loss'] = loss

    # Convert to numpy-friendly arrays for sklearn where needed
    # Keep torch tensors for internal uses below but sklearn can accept torch arrays in many cases.
    # ROC
    fpr, tpr, thresholds = roc_curve(targets, outs, pos_label=1)
    scores['ROC-AUC'] = auc(fpr, tpr)
    scores['Optimal Threshold from ROC'] = thresholds[np.argmax(tpr - fpr)]

    # Precision-Recall
    precision, recall, pr_thresholds = precision_recall_curve(targets, outs)
    # F1 best estimate (small epsilon for numerical stability)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
    scores['F1 Score'] = np.nanmax(f1_scores)
    # Guard pr_thresholds indexing: if pr_thresholds is empty, default to ROC threshold
    try:
        scores['Optimal Threshold from F1'] = pr_thresholds[np.nanargmax(f1_scores)]
    except Exception:
        scores['Optimal Threshold from F1'] = scores['Optimal Threshold from ROC']
    scores['Average Precision'] = average_precision_score(targets, outs)

    # Now use threshold to make predictions and score
    preds = torch.zeros_like(outs)
    match threshold_type.lower():
        case 'roc':
            preds[outs > scores['Optimal Threshold from ROC']] = 1
            scores['Accuracy at Threshold'] = accuracy_score(targets, preds)
            scores['Balanced Accuracy at Threshold'] = balanced_accuracy_score(targets, preds)
            scores['Confusion Matrix'] = confusion_matrix(targets, preds)
            if make_plot:
                fig_out = make_the_plots()
        case 'f1':
            preds[outs > scores['Optimal Threshold from F1']] = 1
            scores['Accuracy at Threshold'] = accuracy_score(targets, preds)
            scores['Balanced Accuracy at Threshold'] = balanced_accuracy_score(targets, preds)
            scores['Confusion Matrix'] = confusion_matrix(targets, preds)
            if make_plot:
                fig_out = make_the_plots()
        case 'none' | 'both':
            fig_out = {}
            # ROC
            preds[outs > scores['Optimal Threshold from ROC']] = 1
            scores['Accuracy at ROC Threshold'] = accuracy_score(targets, preds)
            scores['Balanced Accuracy at ROC Threshold'] = balanced_accuracy_score(targets, preds)
            scores['Confusion Matrix from ROC Threshold'] = confusion_matrix(targets, preds)
            if make_plot:
                fig_out['ROC Threshold'] = make_the_plots()
            # F1
            preds[outs > scores['Optimal Threshold from F1']] = 1
            scores['Accuracy at F1 Threshold'] = accuracy_score(targets, preds)
            scores['Balanced Accuracy at F1 Threshold'] = balanced_accuracy_score(targets, preds)
            scores['Confusion Matrix from F1 Threshold'] = confusion_matrix(targets, preds)
            if make_plot:
                fig_out['F1 Threshold'] = make_the_plots()
        case _:
            raise ValueError(f'Unrecognized threshold type: {threshold_type}. '
                             f'Accepted thresholds are ROC, F1, or both (default).')

    if print_results:
        print('_____________________________________________________')
        for key, item in scores.items():
            if 'Confusion' not in key:
                print(f'|\t{key:<35} {f'{item:.4f}':>10}\t|')
        print('_____________________________________________________')

    if make_plot:
        return scores, fig_out
    else:
        return scores
