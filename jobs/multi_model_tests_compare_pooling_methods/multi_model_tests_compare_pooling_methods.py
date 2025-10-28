# multi_model_tests_with_StageI_all_poolings.py
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

from sklearn.metrics import roc_auc_score

from my_modules.models.classifier_models import *
from my_modules.scripts.model_metrics import score_model
from my_modules.scripts.helper_functions import set_seed
from my_modules.scripts.dataset import NSCLCDataset

# Pooling methods to evaluate
POOL_METHODS = ['min', 'max', 'mean', 'median', 'bottom15', 'top15', 'softmin', 'majority']
# Defaults / params
DEFAULT_POOL_METHOD = 'bottom15'
BOTTOM_FRAC = 0.15
SOFTMIN_TEMP = 10.0
MAJORITY_THRESHOLD = 0.5  # threshold per-image for majority voting


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
        # softmin weighting: lower arr -> higher weight. compute normalized weights.
        # weights = exp(-temp * arr); output = sum(weights * arr) / sum(weights)
        exps = np.exp(-softmin_temp * arr)
        denom = np.sum(exps) + 1e-12
        weights = exps / denom
        return float(np.sum(weights * arr))
    if method == 'majority':
        # fraction of images predicted positive (>maj_thresh)
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
    Returns: (scores_tensor, labels_tensor)
    """
    model.eval()
    patient_scores = []
    patient_labels = []
    image_outputs_log = []  # will hold per-patient dicts

    with torch.no_grad():
        for pt_idx in patient_indices:
            img_indices = dataset.get_patient_subset(pt_idx)
            outs = []
            for im_idx in img_indices:
                x, _ = dataset[im_idx]
                x = x.unsqueeze(0).to(device)
                out = model(x)
                # squeeze to scalar (handles shapes (1,1) or (1,))
                val = out.cpu().detach().squeeze().item()
                outs.append(float(val))
            if len(outs) == 0:
                continue

            score = pool_patient_scores(outs, method=pool_method)
            label = float(dataset.get_patient_label(pt_idx).item())

            patient_scores.append(score)
            patient_labels.append(label)

            # store for CSV
            image_outputs_log.append({
                'patient_index': int(pt_idx),
                'patient_name': str(dataset.get_patient_name(pt_idx)),
                'label': int(label),
                'n_images': len(outs),
                'image_outputs': ';'.join([f"{v:.6f}" for v in outs]),
                'pooled_score': f"{score:.6f}",
                'pool_method': pool_method
            })

            # Diagnostic prints in FAST_TEST
            if FAST_TEST:
                print("---------------------------------")
                print("-----INDIVIDUAL IMAGE OUTPUTS-----")
                print(f"Patient Index: {pt_idx}")
                print(f"Image Outputs: {outs}")
                print(f"Patient Score ({pool_method}): {score}")
                print(f"Patient Label: {int(label)}")
                print("---------------------------------")

    # Save CSV log
    if save_csv:
        # model.name may not exist before model is moved; guard
        model_name = getattr(model, 'name', 'model')
        out_dir = f"outputs/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"image_outputs_patientwise_pool-{pool_method}.csv")
        df = pd.DataFrame(image_outputs_log)
        df.to_csv(csv_path, index=False)

    return torch.tensor(patient_scores), torch.tensor(patient_labels)


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

    # Testing: run all pooling methods and compare
    headers = ['Pooling Method', 'ROC-AUC', 'Optimal Threshold', 'Accuracy at Threshold']
    overall_results = []

    for i, model in enumerate(models):
        print(f'\n>>> {model.name} patient-wise testing across pooling methods...')
        # safe load best
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
            scores_pt, labels_pt = patient_wise_loader_outputs(model, eval_test_data, test_pts, device,
                                                               pool_method=method, save_csv=True)
            # Protect against empty
            if len(scores_pt) == 0:
                print(f'No patient scores returned for pool method {method}. Skipping.')
                continue

            # Score and plot (score_model will compute ROC threshold)
            scores_dict, fig = score_model(model, (scores_pt, labels_pt),
                                           print_results=True, make_plot=True, threshold_type='roc')
            # Extract key metrics
            auc_val = scores_dict.get('ROC-AUC', float('nan'))
            opt_thresh = scores_dict.get('Optimal Threshold from ROC', float('nan'))
            # Accuracy key name depends on score_model internals. Try common names.
            acc_keys = [k for k in scores_dict.keys() if 'Accuracy' in k]
            acc_val = scores_dict.get(acc_keys[0], float('nan')) if len(acc_keys) > 0 else float('nan')

            # Save ROC figure per method
            fname = f'outputs/{model.name}/plots/patientwise_{model.name}_pool-{method}.png'
            try:
                fig.savefig(fname)
                plt.close(fig)
            except Exception:
                # fig might be dict (if both thresholds created); handle cases
                try:
                    # If dict, pick ROC Threshold figure or first item
                    if isinstance(fig, dict):
                        first_fig = list(fig.values())[0]
                        first_fig.savefig(fname)
                        plt.close(first_fig)
                except Exception:
                    pass

            # Append text log
            with open(f'outputs/{model.name}/results.txt', 'a') as f:
                f.write(f'\n>>> {model.name} patient-wise test (pool={method})\n')
                for key, item in scores_dict.items():
                    if 'Confusion' not in key:
                        f.write(f'|\t{key:<35} {format_metric(item):>10}\t|\n')
                f.write('----------------------------------------------------\n')

            per_model_results.append({
                'model': model.name,
                'pool_method': method,
                'roc_auc': auc_val,
                'opt_threshold': float(opt_thresh),
                'accuracy': float(acc_val)
            })

        # Save per-model pooling summary CSV
        if len(per_model_results) > 0:
            summary_df = pd.DataFrame(per_model_results)
            summary_df.to_csv(f'outputs/{model.name}/pooling_method_summary.csv', index=False)
            overall_results.extend(per_model_results)

            # Plot AUC by pooling method
            try:
                fig2, ax = plt.subplots(figsize=(8, 4))
                ax.bar(summary_df['pool_method'], summary_df['roc_auc'])
                ax.set_xlabel('Pooling Method')
                ax.set_ylabel('ROC-AUC')
                ax.set_title(f'{model.name} - ROC-AUC by pooling method')
                plt.xticks(rotation=45)
                fig2.tight_layout()
                fig2.savefig(f'outputs/{model.name}/plots/pooling_method_auc_comparison.png')
                plt.close(fig2)
            except Exception:
                pass

    # Save overall comparison across models (if multiple)
    if len(overall_results) > 0:
        all_df = pd.DataFrame(overall_results)
        all_df.to_csv('outputs/pooling_method_comparison_all_models.csv', index=False)

    # Minimal train-only plots (saved earlier)
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
