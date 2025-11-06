# stageII_splits_min_median.py
# Runs N random balanced Stage II splits. Train once per split, test twice (min + median).
# Uses ROC-optimized threshold per set inside score_model. Only testing metrics reported.

FAST_TEST = False
NUM_SPLITS = 10          # FAST_TEST -> 1
RANDOM_SEED = 42

import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms.v2 as tvt
from matplotlib import pyplot as plt

from my_modules.models.classifier_models import *
from my_modules.scripts.model_metrics import score_model
from my_modules.scripts.helper_functions import set_seed
from my_modules.scripts.dataset import NSCLCDataset

# -----------------------------
# Config
# -----------------------------
POOL_METHODS = ['min', 'median']      # Evaluate both per split
FAST_BATCH = 8
BATCH = 64

def pool_patient_score(outs, method):
    arr = np.asarray(outs, dtype=float)
    if arr.size == 0:
        return float('nan')
    if method == 'min':
        return float(np.min(arr))
    if method == 'median':
        return float(np.median(arr))
    return float(np.median(arr))

def patient_wise_loader_outputs(model, dataset, patient_indices, device, pool_method, save_csv_path=None):
    """Return patient-level pooled scores and labels for given pt indices."""
    model.eval()
    patient_scores, patient_labels, patient_order = [], [], []
    rows = []
    with torch.no_grad():
        for pt_idx in patient_indices:
            img_indices = dataset.get_patient_subset(pt_idx)
            outs = []
            for im_idx in img_indices:
                x, _ = dataset[im_idx]
                x = x.unsqueeze(0).to(device)
                y = model(x).detach().cpu().squeeze().item()
                outs.append(float(y))
            if len(outs) == 0:
                continue
            pooled = pool_patient_score(outs, pool_method)
            label = int(dataset.get_patient_label(pt_idx).item())
            patient_scores.append(pooled)
            patient_labels.append(label)
            patient_order.append(int(pt_idx))
            rows.append({
                'patient_index': int(pt_idx),
                'patient_name': str(dataset.get_patient_name(pt_idx)),
                'label': label,
                'n_images': len(outs),
                'image_outputs': ';'.join([f'{v:.6f}' for v in outs]),
                'pool_method': pool_method,
                'pooled_score': f'{pooled:.6f}'
            })
    if save_csv_path is not None:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        pd.DataFrame(rows).to_csv(save_csv_path, index=False)
    return torch.tensor(patient_scores), torch.tensor(patient_labels), patient_order

def format_metric(x):
    try:
        if torch.is_tensor(x):
            x = x.item() if x.numel() == 1 else x
        if isinstance(x, (int, float, np.integer, np.floating)):
            return f'{float(x):.4f}'
        return str(x)
    except Exception:
        return str(x)

def train_one_model(train_loader, model, device, total_epochs, lr):
    """Train from scratch. Save best by lowest train loss. Return path to best model."""
    loss_fn = torch.nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    best_loss = float('inf')
    best_path = None

    for ep in range(total_epochs):
        model.train()
        running = 0.0
        n = 0
        for x, target in train_loader:
            x = x.to(device)
            target = target.to(device)
            out = model(x)
            loss = loss_fn(out, target.unsqueeze(1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item()) * x.shape[0]
            n += x.shape[0]
        epoch_loss = running / max(1, n)
        print(f'Epoch {ep+1}/{total_epochs} - train loss {epoch_loss:.6f}')
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_dir = f'outputs/{model.name}/models'
            os.makedirs(best_dir, exist_ok=True)
            best_path = os.path.join(best_dir, f'Best_{model.name}.pth')
            torch.save(model.state_dict(), best_path)
    return best_path

def evaluate_and_save(model, dataset, pt_indices, device, pool_method, out_dir):
    """Pool, score_model with ROC threshold, save combined plot and metrics. Return dict with AUCs."""
    # pool
    pooled_csv = os.path.join(out_dir, f'image_outputs_pool-{pool_method}.csv')
    scores_pt, labels_pt, pt_order = patient_wise_loader_outputs(
        model, dataset, pt_indices, device, pool_method, save_csv_path=pooled_csv
    )
    if len(scores_pt) == 0:
        return {'ROC-AUC': float('nan'), 'PR-AUC': float('nan')}

    # score with ROC-optimized threshold and plots
    scores_dict, fig = score_model(
        model, (scores_pt, labels_pt),
        print_results=False, make_plot=True,
        threshold_type='roc'
    )
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'combined_plot.png'))
    plt.close(fig)

    # write metrics
    cm = scores_dict.get('Confusion Matrix', None)
    acc = scores_dict.get('Accuracy', None)
    bal_acc = scores_dict.get('Balanced Accuracy', None)
    roc_auc = scores_dict.get('ROC-AUC', float('nan'))
    pr_auc = scores_dict.get('Average Precision', float('nan'))
    thr = scores_dict.get('ROC-Optimal Threshold', None)

    # save pooled scores for replication
    pd.DataFrame({
        'patient_index': pt_order,
        'pooled_score': scores_pt.numpy(),
        'label': labels_pt.numpy()
    }).to_csv(os.path.join(out_dir, f'pooled_scores_{pool_method}.csv'), index=False)

    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
        f.write(f'Model: {model.name}\n')
        f.write(f'Pooling: {pool_method}\n\n')
        f.write(f'ROC-AUC: {format_metric(roc_auc)}\n')
        f.write(f'PR-AUC: {format_metric(pr_auc)}\n')
        if thr is not None:
            f.write(f'ROC-Optimal Threshold: {format_metric(thr)}\n')
        if acc is not None:
            f.write(f'Accuracy: {format_metric(acc)}\n')
        if bal_acc is not None:
            f.write(f'Balanced Accuracy: {format_metric(bal_acc)}\n')
        if cm is not None:
            f.write(f'Confusion Matrix (rows=true, cols=pred):\n{cm}\n')

    return {'ROC-AUC': float(roc_auc), 'PR-AUC': float(pr_auc)}

def main():
    # global seeds
    set_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f'Num cores: {mp.cpu_count()}')
    print(f'Num GPUs: {torch.cuda.device_count()}')
    try:
        mp.set_start_method('forkserver', force=True)
    except RuntimeError:
        pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # datasets
    ds_train = NSCLCDataset('NSCLC_Data_for_ML', ['fad','nadh','shg','intensity','orr'],
                            device=torch.device('cpu'), label='Metastases', mask_on=True)
    ds_eval = NSCLCDataset('NSCLC_Data_for_ML', ['fad','nadh','shg','intensity','orr'],
                           device=torch.device('cpu'), label='Metastases', mask_on=True)

    if FAST_TEST:
        ds_train.augmented = False
        ds_train.augment_patients = False
        ds_eval.augmented = False
        ds_eval.augment_patients = False
        ds_train.normalize_method = 'preset'
        ds_eval.normalize_method = 'preset'
        ds_train.transforms = None
        ds_eval.transforms = None
        ds_train.to(device)
        ds_eval.to(device)
    else:
        ds_train.augment()
        ds_train.normalize_method = 'preset'
        ds_train.to(device)
        ds_train.transforms = tvt.Compose([
            tvt.RandomVerticalFlip(p=0.25),
            tvt.RandomHorizontalFlip(p=0.25),
            tvt.RandomRotation(degrees=(-180, 180))
        ])
        ds_eval.augment()
        ds_eval.normalize_method = 'preset'
        ds_eval.to(device)

    # enumerate patients and filter Stage II only
    subsampler = torch.utils.data.sampler.SubsetRandomSampler(range(ds_train.patient_count))
    all_idx = [i for i in subsampler]
    # drop empty
    empty = [i for i in all_idx if len(ds_train.get_patient_subset(i)) == 0]
    for e in empty:
        all_idx.remove(e)

    stageII = []
    patient_names = {}
    for i in all_idx:
        name = ds_train.get_patient_name(i)
        patient_names[i] = name
        # Stage II = not ending with '_StageI'
        if not (isinstance(name, str) and name.endswith('_StageI')):
            stageII.append(i)

    if len(stageII) < 25:
        print(f'Warning: detected {len(stageII)} Stage II patients (< 25). Proceeding with available.')
    labels_stageII = [int(ds_train.get_patient_label(i).item()) for i in stageII]
    pairs = list(zip(stageII, labels_stageII))

    # split class lists
    zeros = [i for i, l in pairs if l == 0]  # metastatic
    ones  = [i for i, l in pairs if l == 1]  # non-metastatic

    if len(zeros) < 4 or len(ones) < 4:
        raise RuntimeError('Not enough Stage II patients to form balanced test sets (4 met + 4 non-met).')

    # runtime knobs
    if FAST_TEST:
        total_splits = 1
        epochs = 1
        lr = 1e-4
        batch_size = FAST_BATCH
    else:
        total_splits = NUM_SPLITS
        epochs = 2500
        lr = 1e-8
        batch_size = BATCH

    # prep outputs
    os.makedirs('outputs', exist_ok=True)
    split_rows = []
    combo_rows = []

    for split_id in range(total_splits):
        print('\n' + '='*70)
        print(f'Split {split_id+1}/{total_splits}')

        # different randomness per split but deterministic
        rnd = random.Random(RANDOM_SEED + split_id)

        # sample balanced test: 4 non-met + 4 met
        test_nonmet = rnd.sample(ones, 4)
        test_met = rnd.sample(zeros, 4)
        test_pts = test_nonmet + test_met
        train_pts = [pt for pt in stageII if pt not in test_pts]

        # optional truncation in FAST_TEST
        if FAST_TEST:
            train_pts = train_pts[:1]
            test_pts = test_pts[:1]

        # report split
        tn = sum(1 for i in train_pts if int(ds_train.get_patient_label(i).item()) == 1)
        tm = sum(1 for i in train_pts if int(ds_train.get_patient_label(i).item()) == 0)
        sn = sum(1 for i in test_pts  if int(ds_train.get_patient_label(i).item()) == 1)
        sm = sum(1 for i in test_pts  if int(ds_train.get_patient_label(i).item()) == 0)
        print(f'Train: {len(train_pts)} pts -> {tn} non-met, {tm} met')
        print(f'Test : {len(test_pts)} pts -> {sn} non-met, {sm} met')

        # save split combination row
        combo_rows.append({
            'split_id': split_id,
            'train_patient_indices': ';'.join(map(str, train_pts)),
            'train_patient_names': ';'.join(str(patient_names[i]) for i in train_pts),
            'test_patient_indices': ';'.join(map(str, test_pts)),
            'test_patient_names': ';'.join(str(patient_names[i]) for i in test_pts),
            'train_nonmet': tn, 'train_met': tm,
            'test_nonmet': sn, 'test_met': sm
        })

        # build loaders
        train_idx = [ds_train.get_patient_subset(i) for i in train_pts]
        train_idx = [im for sub in train_idx for im in sub]
        rnd.shuffle(train_idx)

        test_idx = [ds_eval.get_patient_subset(i) for i in test_pts]
        test_idx = [im for sub in test_idx for im in sub]
        rnd.shuffle(test_idx)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds_train, train_idx),
            batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_set = torch.utils.data.Subset(ds_eval, test_idx)

        # model fresh init
        model = ResNet18NPlaned(ds_train.shape, start_width=64, n_classes=1)
        if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
            model.to(device)

        # train and get best path
        best_path = train_one_model(train_loader, model, device, total_epochs=epochs, lr=lr)
        if best_path is not None:
            model.load_state_dict(torch.load(best_path))

        # evaluate both pooling methods sequentially
        for pool in POOL_METHODS:
            out_dir = f'outputs/{model.name}/split_{split_id}_{pool}'
            os.makedirs(out_dir, exist_ok=True)

            # patient-wise evaluation on test set
            metrics = evaluate_and_save(
                model, ds_eval, test_pts, device, pool_method=pool, out_dir=out_dir
            )
            roc_auc = metrics.get('ROC-AUC', float('nan'))
            pr_auc  = metrics.get('PR-AUC', float('nan'))

            # record split summary row
            split_rows.append({
                'split_id': split_id,
                'model': model.name,
                'pool_method': pool,
                'test_nonmet': sn, 'test_met': sm,
                'train_nonmet': tn, 'train_met': tm,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'best_model_path': best_path
            })
            print(f'[{model.name} | {pool}] ROC-AUC={format_metric(roc_auc)}  PR-AUC={format_metric(pr_auc)}')

    # write summaries
    pd.DataFrame(split_rows).to_csv('outputs/split_results.csv', index=False)
    pd.DataFrame(combo_rows).to_csv('outputs/split_combinations.csv', index=False)
    print('\nDone. Wrote:')
    print(' - outputs/split_results.csv')
    print(' - outputs/split_combinations.csv')

if __name__ == '__main__':
    main()
