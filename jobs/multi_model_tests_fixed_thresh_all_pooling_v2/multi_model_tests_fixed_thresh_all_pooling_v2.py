# multi_model_tests_with_StageI_all_poolings_fixed_threshold_plots_A2.py
# Toggle FAST_TEST to switch between smoke test and full run
FAST_TEST = False

import os
import math
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms.v2 as tvt
from matplotlib import pyplot as plt
import random
from datetime import datetime

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from my_modules.models.classifier_models import *
from my_modules.scripts.helper_functions import set_seed
from my_modules.scripts.dataset import NSCLCDataset

# Pooling methods to evaluate
POOL_METHODS = ['min', 'max', 'mean', 'median', 'bottom15', 'top15', 'softmin', 'majority']
# Defaults / params
DEFAULT_POOL_METHOD = 'bottom15'
BOTTOM_FRAC = 0.15
SOFTMIN_TEMP = 10.0
MAJORITY_THRESHOLD = 0.5  # per-image threshold used for majority voting
TEST_THRESHOLD = 0.5      # fixed test threshold requested


def format_metric(item):
    import torch as _torch
    try:
        if _torch.is_tensor(item):
            if item.numel() == 1:
                return f"{float(item.item()):.4f}"
            return str(item)
        if isinstance(item, (int, float, np.floating, np.integer)):
            return f"{float(item):.4f}"
        return f"{float(item):.4f}"
    except Exception:
        return str(item)


def pool_patient_scores(outs, method=DEFAULT_POOL_METHOD, bottom_frac=BOTTOM_FRAC,
                        softmin_temp=SOFTMIN_TEMP, maj_thresh=MAJORITY_THRESHOLD):
    """
    outs: list/1D-array of per-image model outputs (higher => more non-metastatic).
    Returns a scalar score in [0,1] for patient.
    """
    if len(outs) == 0:
        return float('nan')
    arr = np.array(outs, dtype=float)

    method = method.lower()
    if method == 'min':
        return float(np.min(arr))
    if method == 'max':
        return float(np.max(arr))
    if method == 'mean':
        return float(np.mean(arr))
    if method == 'median':
        return float(np.median(arr))
    if method == 'bottom15':
        k = max(1, int(np.ceil(len(arr) * bottom_frac)))
        s = np.sort(arr)[:k]
        return float(np.mean(s))
    if method == 'top15':
        k = max(1, int(np.ceil(len(arr) * bottom_frac)))
        s = np.sort(arr)[-k:]
        return float(np.mean(s))
    if method == 'softmin':
        exps = np.exp(-softmin_temp * arr)
        denom = np.sum(exps) + 1e-12
        weights = exps / denom
        return float(np.sum(weights * arr))
    if method == 'majority':
        preds = (arr > maj_thresh).astype(int)
        frac = preds.sum() / len(preds)
        return float(frac)
    # fallback
    return float(np.median(arr))


def patient_wise_loader_outputs(model, dataset, patient_indices, device,
                                pool_method=DEFAULT_POOL_METHOD, save_csv=True):
    """
    Run model on all images for each patient in patient_indices and pool using pool_method.
    Saves CSV of per-image outputs (one CSV per model & pooling method).
    Returns: (scores_tensor, labels_tensor, patient_index_order_list)
    """
    model.eval()
    patient_scores = []
    patient_labels = []
    image_outputs_log = []
    patient_order = []

    with torch.no_grad():
        for pt_idx in patient_indices:
            img_indices = dataset.get_patient_subset(pt_idx)
            outs = []
            for im_idx in img_indices:
                x, _ = dataset[im_idx]
                x = x.unsqueeze(0).to(device)
                out = model(x)
                val = out.cpu().detach().squeeze().item()
                outs.append(float(val))
            if len(outs) == 0:
                continue

            score = pool_patient_scores(outs, method=pool_method)
            label = float(dataset.get_patient_label(pt_idx).item())

            patient_scores.append(score)
            patient_labels.append(label)
            patient_order.append(int(pt_idx))

            image_outputs_log.append({
                'patient_index': int(pt_idx),
                'patient_name': str(dataset.get_patient_name(pt_idx)),
                'label': int(label),
                'n_images': len(outs),
                'image_outputs': ';'.join([f"{v:.6f}" for v in outs]),
                'pooled_score': f"{score:.6f}",
                'pool_method': pool_method
            })

            if FAST_TEST:
                print("---------------------------------")
                print("-----INDIVIDUAL IMAGE OUTPUTS-----")
                print(f"Patient Index: {pt_idx}")
                print(f"Image Outputs: {outs}")
                print(f"Patient Score ({pool_method}): {score}")
                print(f"Patient Label: {int(label)}")
                print("---------------------------------")

    if save_csv:
        model_name = getattr(model, 'name', 'model')
        out_dir = f"outputs/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"image_outputs_patientwise_pool-{pool_method}.csv")
        try:
            df = pd.DataFrame(image_outputs_log)
            df.to_csv(csv_path, index=False)
        except Exception:
            pass

    return torch.tensor(patient_scores), torch.tensor(patient_labels), patient_order


def save_plots_and_metrics(model_name, method, labels_np, scores_np, test_patient_indices_order,
                           stageI_pts, out_dir_pool, threshold=TEST_THRESHOLD):
    """
    Create ROC, PR, and Confusion Matrix @ threshold=0.5 and write metrics file.
    out_dir_pool should exist.
    """
    metrics = {}
    # ROC AUC
    try:
        roc_auc = float(roc_auc_score(labels_np, scores_np))
        metrics['roc_auc'] = roc_auc
        fpr, tpr, roc_thresh = roc_curve(labels_np, scores_np)
    except Exception:
        roc_auc = float('nan')
        fpr, tpr, roc_thresh = None, None, None
        metrics['roc_auc'] = float('nan')

    # PR AUC
    try:
        precision, recall, pr_thresh = precision_recall_curve(labels_np, scores_np)
        pr_auc = float(average_precision_score(labels_np, scores_np))
        metrics['pr_auc'] = pr_auc
    except Exception:
        precision, recall, pr_thresh = None, None, None
        metrics['pr_auc'] = float('nan')

    # Save ROC plot
    try:
        if fpr is not None and tpr is not None:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
            ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{model_name} - ROC ({method})')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir_pool, 'roc.png'))
            plt.close(fig)
    except Exception:
        pass

    # Save PR plot
    try:
        if precision is not None and recall is not None:
            fig, ax = plt.subplots(figsize=(6, 5))
            PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax=ax)
            ax.set_title(f'{model_name} - Precision-Recall ({method})')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir_pool, 'pr.png'))
            plt.close(fig)
    except Exception:
        pass

    # Predictions at fixed threshold
    preds = (scores_np > threshold).astype(int)
    labels_int = labels_np.astype(int)

    # Confusion matrix at threshold
    try:
        cm = confusion_matrix(labels_int, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4))
        disp.plot(ax=ax, values_format='d', cmap='Blues')
        ax.set_title(f'{model_name} - Confusion (@{threshold}) ({method})')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir_pool, 'confusion_matrix.png'))
        plt.close(fig)
    except Exception:
        pass

    # Stage masks and accuracies
    # test_patient_indices_order is the list of patient indices in the order scores_np corresponds to.
    # Build masks from that list.
    patient_indices = test_patient_indices_order
    patient_indices = [int(x) for x in patient_indices]
    stageI_mask = np.array([1 if p in stageI_pts else 0 for p in patient_indices], dtype=bool)
    stageII_mask = ~stageI_mask

    def acc_counts(mask):
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return (0, 0, float('nan'))
        correct = int((preds[idxs] == labels_int[idxs]).sum())
        total = len(idxs)
        acc = correct / total
        return (correct, total, acc)

    overall_correct = int((preds == labels_int).sum())
    overall_total = len(labels_int)
    overall_acc = overall_correct / overall_total if overall_total > 0 else float('nan')
    s1_correct, s1_total, s1_acc = acc_counts(stageI_mask)
    s2_correct, s2_total, s2_acc = acc_counts(stageII_mask)

    metrics.update({
        'overall_correct': int(overall_correct),
        'overall_total': int(overall_total),
        'overall_acc': float(overall_acc),
        'stageI_correct': int(s1_correct),
        'stageI_total': int(s1_total),
        'stageI_acc': float(s1_acc) if not np.isnan(s1_acc) else float('nan'),
        'stageII_correct': int(s2_correct),
        'stageII_total': int(s2_total),
        'stageII_acc': float(s2_acc) if not np.isnan(s2_acc) else float('nan')
    })

    # Write metrics.txt
    metrics_path = os.path.join(out_dir_pool, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Pooling method: {method}\n")
        f.write(f"Fixed threshold: {threshold}\n\n")
        f.write(f"ROC-AUC: {metrics['roc_auc']:.6f}\n")
        f.write(f"PR-AUC (Average Precision): {metrics['pr_auc']:.6f}\n\n")
        f.write(f"Overall: {metrics['overall_correct']}/{metrics['overall_total']} correct -> {metrics['overall_acc']:.6f}\n")
        f.write(f"Stage I: {metrics['stageI_correct']}/{metrics['stageI_total']} correct -> {metrics['stageI_acc'] if not math.isnan(metrics['stageI_acc']) else 'N/A'}\n")
        f.write(f"Stage II: {metrics['stageII_correct']}/{metrics['stageII_total']} correct -> {metrics['stageII_acc'] if not math.isnan(metrics['stageII_acc']) else 'N/A'}\n")

    return metrics


def main():
    set_seed(42)
    random.seed(42)
    np.random.seed(42)

    print(f'Num cores: {mp.cpu_count()}')
    print(f'Num GPUs: {torch.cuda.device_count()}')
    try:
        mp.set_start_method('forkserver', force=True)
    except RuntimeError:
        pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ################
    # Prepare data #
    ################
    train_data = NSCLCDataset('NSCLC_Data_for_ML', ['fad', 'nadh', 'shg', 'intensity', 'orr'],
                              device=torch.device('cpu'), label='Metastases', mask_on=True)
    eval_test_data = NSCLCDataset('NSCLC_Data_for_ML', ['fad', 'nadh', 'shg', 'intensity', 'orr'],
                                  device=torch.device('cpu'), label='Metastases', mask_on=True)

    if FAST_TEST:
        train_data.augmented = False
        train_data.augment_patients = False
        eval_test_data.augmented = False
        eval_test_data.augment_patients = False
        train_data.normalize_method = 'preset'
        eval_test_data.normalize_method = 'preset'
        train_data.transforms = None
        eval_test_data.transforms = None
        train_data.to(device)
        eval_test_data.to(device)
    else:
        train_data.augment()
        train_data.normalize_method = 'preset'
        train_data.to(device)
        train_data.transforms = tvt.Compose([tvt.RandomVerticalFlip(p=0.25),
                                             tvt.RandomHorizontalFlip(p=0.25),
                                             tvt.RandomRotation(degrees=(-180, 180))])
        eval_test_data.augment()
        eval_test_data.normalize_method = 'preset'
        eval_test_data.to(device)

    # Build initial patient index list
    subsampler = torch.utils.data.sampler.SubsetRandomSampler(range(train_data.patient_count))
    idx = [i for i in subsampler]

    # Remove empty patients
    patient_subsets = [train_data.get_patient_subset(i) for i in idx]
    idx_for_removal = [idx[i] for i, subset in enumerate(patient_subsets) if len(subset) == 0]
    for ix in idx_for_removal:
        idx.remove(ix)

    # Stage I vs Stage II
    stageI_pts = []
    stageII_pts = []
    patient_names = {}
    for i in idx:
        name = train_data.get_patient_name(i)
        patient_names[i] = name
        if isinstance(name, str) and name.endswith('_StageI'):
            stageI_pts.append(i)
        else:
            stageII_pts.append(i)

    print(f'Found {len(stageII_pts)} Stage II patients and {len(stageI_pts)} Stage I patients.')
    print('Stage I patient names (Stage I => TEST set):')
    for i in stageI_pts:
        print(f'  - idx {i}: {patient_names[i]}')
    print('Stage II patient names (to be split into TRAIN / TEST):')
    for i in stageII_pts:
        print(f'  - idx {i}: {patient_names[i]}')

    if len(stageII_pts) == 0:
        raise RuntimeError('No Stage II patients detected.')

    # Stage II labels
    labels_stageII = [train_data.get_patient_label(i).item() for i in stageII_pts]
    paired = list(zip(stageII_pts, labels_stageII))
    random.shuffle(paired)
    zeros = [i for i, l in paired if int(l) == 0]  # metastatic
    ones = [i for i, l in paired if int(l) == 1]   # non-metastatic

    print(f'Stage II metastatic (label=0): {len(zeros)} patients')
    print(f'Stage II non-metastatic (label=1): {len(ones)} patients')

    # Move 3 non-met and 5 met from Stage II to test
    if len(ones) < 3 or len(zeros) < 5:
        raise RuntimeError('Not enough Stage II patients in a class to sample requested test set.')
    test_from_nonmet = random.sample(ones, 3)
    test_from_met = random.sample(zeros, 5)

    train_pts = [pt for pt in stageII_pts if pt not in (test_from_nonmet + test_from_met)]
    test_pts = stageI_pts + test_from_nonmet + test_from_met

    train_nonmet = sum(1 for i in train_pts if int(train_data.get_patient_label(i).item()) == 1)
    train_met = sum(1 for i in train_pts if int(train_data.get_patient_label(i).item()) == 0)
    test_nonmet = sum(1 for i in test_pts if int(train_data.get_patient_label(i).item()) == 1)
    test_met = sum(1 for i in test_pts if int(train_data.get_patient_label(i).item()) == 0)

    print('\nSPLIT COUNTS (after StageII reassignment):')
    print(f'  TRAIN StageII: {len(train_pts)} patients -> {train_nonmet} non-met, {train_met} met')
    print(f'  TEST (StageI + selected StageII): {len(test_pts)} patients -> {test_nonmet} non-met, {test_met} met')

    if FAST_TEST:
        n_per_split = 1
        train_pts = train_pts[:n_per_split]
        test_pts = test_pts[:n_per_split]
        print('FAST_TEST enabled: using first patient from each split only.')

    print('\nFINAL SPLITS (patient indices and names):')
    print(f'  TRAIN (n={len(train_pts)})')
    for i in train_pts:
        print(f'    idx {i}: {patient_names[i]} (label={train_data.get_patient_label(i).item()})')
    print(f'  TEST (n={len(test_pts)})')
    for i in test_pts:
        print(f'    idx {i}: {patient_names[i]} (label={train_data.get_patient_label(i).item()})')

    # Flatten image indices
    train_idx = [train_data.get_patient_subset(i) for i in train_pts]
    train_idx = [im for i in train_idx for im in i]
    random.shuffle(train_idx)

    test_idx = [eval_test_data.get_patient_subset(i) for i in test_pts]
    test_idx = [im for i in test_idx for im in i]
    random.shuffle(test_idx)

    comb_pts = test_pts[:]
    comb_idx = [eval_test_data.get_patient_subset(i) for i in comb_pts]
    comb_idx = [im for i in comb_idx for im in i]
    random.shuffle(comb_idx)

    # Image counts
    train_image_counts = [0, 0]
    for pt in train_pts:
        label = int(train_data.get_patient_label(pt).item())
        train_image_counts[label] += len(train_data.get_patient_subset(pt))

    test_image_counts = [0, 0]
    for pt in test_pts:
        label = int(eval_test_data.get_patient_label(pt).item())
        test_image_counts[label] += len(eval_test_data.get_patient_subset(pt))

    comb_image_counts = [test_image_counts[0], test_image_counts[1]]

    print(f'\nTraining set summary: {len(train_pts)} patients, {len(train_idx)} images. Image counts per class: {train_image_counts}')
    print(f'Test  set summary: {len(test_pts)} patients, {len(test_idx)} images. Image counts per class: {test_image_counts}')
    print(f'Combined Test summary: {len(comb_pts)} patients, {len(comb_idx)} images. Image counts per class: {comb_image_counts}')

    # Dataloaders
    batch_size = 8 if FAST_TEST else 64
    train_set = torch.utils.data.Subset(train_data, train_idx)
    test_set = torch.utils.data.Subset(eval_test_data, test_idx)
    comb_set = torch.utils.data.Subset(eval_test_data, comb_idx)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size, shuffle=True, num_workers=0,
                                               drop_last=(True if len(train_idx) % batch_size == 1 else False))
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size, shuffle=False, num_workers=0,
                                              drop_last=(True if len(test_idx) % batch_size == 1 else False))
    comb_loader = torch.utils.data.DataLoader(comb_set,
                                              batch_size=batch_size, shuffle=False, num_workers=0,
                                              drop_last=(True if len(comb_idx) % batch_size == 1 else False))

    #####################
    # Prepare model zoo #
    #####################
    models = [ResNet18NPlaned(train_data.shape, start_width=64, n_classes=1)]
    if not FAST_TEST:
        models[len(models):] = [CNNet(train_data.shape),
                                RegularizedCNNet(train_data.shape)]
    for model in models:
        if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
            model.to(device)

    ###################
    # Hyperparameters #
    ###################
    if FAST_TEST:
        epochs = [1, 5]
        total_epochs = max(epochs)
        learning_rate = 1e-4
    else:
        epochs = [250, 500, 1500, 2000, 2500]
        total_epochs = epochs[-1]
        learning_rate = 1e-8

    loss_function = nn.BCELoss()
    optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01) for model in models]

    ###############
    # Output Prep #
    ###############
    for model in models:
        os.makedirs(f'outputs/{model.name}/plots', exist_ok=True)
        os.makedirs(f'outputs/{model.name}/models', exist_ok=True)
        with open(f'outputs/{model.name}/results.txt', 'w') as f:
            f.write(f'{model.name} Results\n')
    if not os.path.exists('outputs/results.txt'):
        with open('outputs/results.txt', 'w') as f:
            f.write('Overall Results\n')

    train_loss = [[] for _ in range(len(models))]
    train_auc = [[] for _ in range(len(models))]
    eval_loss = [[] for _ in range(len(models))]   # unused
    eval_auc = [[] for _ in range(len(models))]    # unused
    best_score = [0 for _ in range(len(models))]

    # Training loop (no validation)
    for ep in range(total_epochs):
        print(f'\nEpoch {ep + 1}')

        epoch_loss = [0 for _ in range(len(models))]
        outs = [torch.tensor([]) for _ in range(len(models))]
        targets = [torch.tensor([]) for _ in range(len(models))]

        for model in models:
            model.train()
        for x, target in train_loader:
            x = x.to(device)
            target = target.to(device)
            for i, model in enumerate(models):
                out = model(x)
                outs[i] = torch.cat((outs[i], out.cpu().detach()), dim=0)
                targets[i] = torch.cat((targets[i], target.cpu().detach()), dim=0)
                loss = loss_function(out, target.unsqueeze(1))
                optimizers[i].zero_grad()
                loss.backward()
                epoch_loss[i] += loss.item()
                optimizers[i].step()

        for el, tl, ta, tx, ot, model in zip(epoch_loss, train_loss, train_auc, targets, outs, models):
            tl.append(el / (len(train_set) if len(train_set) > 0 else 1))
            try:
                ta.append(roc_auc_score(tx, ot))
            except Exception:
                ta.append(0.0)

        # Save best by training AUC (no validation)
        for i, (model, tl, ta) in enumerate(zip(models, train_loss, train_auc)):
            train_auc_val = ta[-1] if len(ta) > 0 else 0.0
            print(f'>>> {model.name}: Train - Loss: {tl[-1] if len(tl) > 0 else 0.0:.4f}. AUC: {train_auc_val:.4f}.')
            with open(f'outputs/{model.name}/results.txt', 'a') as f:
                f.write(f'\nEpoch {ep + 1} '
                        f'>>> {model.name}: Train - Loss: {tl[-1] if len(tl) > 0 else 0.0:.4f}. '
                        f'AUC: {train_auc_val:.4f}.')

            if train_auc_val > best_score[i] or FAST_TEST:
                best_score[i] = train_auc_val
                torch.save(model.state_dict(), f'outputs/{model.name}/models/Best {model.name}.pth')
                with open(f'outputs/{model.name}/results.txt', 'a') as f:
                    f.write(f'\nNew best {model.name} saved at epoch {ep + 1} with AUC of {train_auc_val:.4f}')
                with open(f'outputs/results.txt', 'a') as f:
                    f.write(f'\nNew best {model.name} saved at epoch {ep + 1} with AUC of {train_auc_val:.4f}')

            if (ep + 1) in epochs:
                torch.save(model.state_dict(), f'outputs/{model.name}/models/Epochs {ep + 1} {model.name}.pth')

    with open(f'outputs/results.txt', 'a') as f:
        f.write(f'\n\nFinal AUC Results\n')
        for model, bs, ta in zip(models, best_score, train_auc):
            f.write(f'{model.name}: Best train AUC - {bs:.4f}. Final Train AUC - {ta[-1] if len(ta)>0 else 0.0:.4f}\n')

    # Testing: run all pooling methods and compare with fixed threshold, create ROC/PR/CM plots
    overall_results = []

    for i, model in enumerate(models):
        print(f'\n>>> {model.name} patient-wise testing across pooling methods (fixed threshold={TEST_THRESHOLD})...')
        best_model_path = f'outputs/{model.name}/models/Best {model.name}.pth'
        fallback_path = f'outputs/{model.name}/models/Epochs {epochs[0]} {model.name}.pth' if len(epochs) > 0 else None
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
        elif fallback_path and os.path.exists(fallback_path):
            print(f'No Best model found for {model.name}; using {fallback_path}')
            model.load_state_dict(torch.load(fallback_path))
        else:
            print(f'No saved model found for {model.name}. Skipping testing for this model.')
            continue

        per_model_results = []

        for method in POOL_METHODS:
            scores_pt, labels_pt, patient_order = patient_wise_loader_outputs(model, eval_test_data, test_pts, device,
                                                                             pool_method=method, save_csv=True)
            if len(scores_pt) == 0:
                print(f'No patient scores returned for pool method {method}. Skipping.')
                continue

            # Prepare numpy arrays
            scores_np = scores_pt.numpy()
            labels_np = labels_pt.numpy().astype(int)

            # Create pooling-specific output folder under plots/ (A2)
            out_dir_pool = f'outputs/{model.name}/plots/pool_{method}'
            os.makedirs(out_dir_pool, exist_ok=True)

            # Save CSV of per-patient pooled scores also in the pool folder for traceability
            pooled_csv_path = os.path.join(out_dir_pool, f'pooled_scores_{method}.csv')
            try:
                pd.DataFrame({'patient_index': patient_order, 'pooled_score': scores_np, 'label': labels_np}).to_csv(pooled_csv_path, index=False)
            except Exception:
                pass

            # Save plots and metrics (ROC, PR, CM @ 0.5) and return metrics dict
            metrics = save_plots_and_metrics(model.name, method, labels_np, scores_np, patient_order, stageI_pts, out_dir_pool, threshold=TEST_THRESHOLD)

            # Append to per-model results
            per_model_results.append({
                'model': model.name,
                'pool_method': method,
                'roc_auc': metrics.get('roc_auc', float('nan')),
                'pr_auc': metrics.get('pr_auc', float('nan')),
                'overall_correct': metrics.get('overall_correct', 0),
                'overall_total': metrics.get('overall_total', 0),
                'overall_acc': metrics.get('overall_acc', float('nan')),
                'stageI_correct': metrics.get('stageI_correct', 0),
                'stageI_total': metrics.get('stageI_total', 0),
                'stageI_acc': metrics.get('stageI_acc', float('nan')),
                'stageII_correct': metrics.get('stageII_correct', 0),
                'stageII_total': metrics.get('stageII_total', 0),
                'stageII_acc': metrics.get('stageII_acc', float('nan'))
            })

            # Also append to model-level results file
            with open(f'outputs/{model.name}/results.txt', 'a') as f:
                f.write(f'\nPOOL METHOD: {method}\n')
                f.write(f'  ROC-AUC: {metrics.get("roc_auc", float("nan")):.6f}\n')
                f.write(f'  PR-AUC: {metrics.get("pr_auc", float("nan")):.6f}\n')
                f.write(f'  Overall: {metrics.get("overall_correct",0)}/{metrics.get("overall_total",0)} -> {metrics.get("overall_acc",float("nan")):.6f}\n')
                f.write(f'  Stage I: {metrics.get("stageI_correct",0)}/{metrics.get("stageI_total",0)} -> {metrics.get("stageI_acc",float("nan"))}\n')
                f.write(f'  Stage II: {metrics.get("stageII_correct",0)}/{metrics.get("stageII_total",0)} -> {metrics.get("stageII_acc",float("nan"))}\n')
                f.write('----------------------------------------------------\n')

        # Save per-model pooling summary CSV and condensed comparison
        if len(per_model_results) > 0:
            summary_df = pd.DataFrame(per_model_results)
            summary_df.to_csv(f'outputs/{model.name}/pooling_method_summary_fixed_threshold.csv', index=False)
            overall_results.extend(per_model_results)

            # Save concise comparison into plots folder for quick view
            comp_df = summary_df[['pool_method', 'overall_acc', 'stageI_acc', 'stageII_acc', 'roc_auc', 'pr_auc']]
            try:
                comp_df.to_csv(f'outputs/{model.name}/plots/pooling_accuracy_comparison.csv', index=False)
            except Exception:
                pass

            # Print human readable summary to console
            print(f'\n=== Summary table for {model.name} (fixed threshold={TEST_THRESHOLD}) ===')
            print(comp_df.to_string(index=False))
            print('====================================================\n')

    # Save overall comparison across models (if multiple)
    if len(overall_results) > 0:
        all_df = pd.DataFrame(overall_results)
        all_df.to_csv('outputs/pooling_method_comparison_all_models_fixed_threshold.csv', index=False)
        human_cols = ['model', 'pool_method', 'roc_auc', 'pr_auc', 'overall_correct', 'overall_total', 'overall_acc',
                      'stageI_correct', 'stageI_total', 'stageI_acc', 'stageII_correct', 'stageII_total', 'stageII_acc']
        try:
            all_df[human_cols].to_csv('outputs/pooling_method_accuracy_readout.csv', index=False)
        except Exception:
            all_df.to_csv('outputs/pooling_method_accuracy_readout.csv', index=False)

    # Minimal train-only plots saved earlier (unchanged)
    headers = ['Training Loss (average per sample)', 'Training ROC-AUC']
    for (model, tl, ta) in zip(models, train_loss, train_auc):
        outputs = [[a, c] for (a, c) in zip(tl, ta)]
        output_table = pd.DataFrame(data=outputs, index=range(1, total_epochs + 1), columns=headers)
        output_table.to_csv(f'outputs/{model.name}/tabular.csv', index_label='Epoch')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        plt.suptitle(model.name)
        ax1.plot(range(1, total_epochs + 1), tl, label=f'{model.name} Training')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax2.plot(range(1, total_epochs + 1), ta, label=f'{model.name} Training AUC')
        ax2.set_ylabel('AUC')
        ax2.set_title('Training ROC-AUC')
        ax2.legend()
        fig.savefig(f'outputs/{model.name}/plots/losses_and_aucs.png')
        plt.close(fig)


if __name__ == '__main__':
    main()
