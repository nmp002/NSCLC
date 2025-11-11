# fixed_stageII_split_trainThreshold_confusion.py

FAST_TEST = False
RANDOM_SEED = 42

import os
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

POOL_METHODS = ['min', 'median']
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
        print(f'Epoch {ep + 1}/{total_epochs} - train loss {epoch_loss:.6f}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_dir = f'outputs/{model.name}/models'
            os.makedirs(best_dir, exist_ok=True)
            best_path = os.path.join(best_dir, f'Best_{model.name}.pth')
            torch.save(model.state_dict(), best_path)

    return best_path


def evaluate_fixed_threshold(model, scores_pt, labels_pt, threshold, out_dir, label_tag):
    os.makedirs(out_dir, exist_ok=True)
    scores_dict, fig = score_model(
        model,
        (scores_pt, labels_pt),
        print_results=False,
        make_plot=True,
        threshold_type='fixed',
        threshold=float(threshold)
    )
    fig.savefig(os.path.join(out_dir, f'combined_plot_{label_tag}.png'))
    plt.close(fig)

    with open(os.path.join(out_dir, f'metrics_{label_tag}.txt'), 'w') as f:
        f.write(f'Model: {model.name}\n')
        f.write(f'Set: {label_tag}\n')
        f.write(f'Fixed threshold from training: {format_metric(threshold)}\n\n')
        for k, v in scores_dict.items():
            if 'Confusion' in k:
                f.write(f'\n{k}:\n{v}\n')
            elif any(w in k for w in ['Accuracy', 'Precision', 'Recall', 'F1']):
                f.write(f'{k}: {format_metric(v)}\n')

    return scores_dict


def main():
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

    ds_train = NSCLCDataset('NSCLC_Data_for_ML', ['fad', 'nadh', 'shg', 'intensity', 'orr'],
                            device=torch.device('cpu'), label='Metastases', mask_on=True)
    ds_eval = NSCLCDataset('NSCLC_Data_for_ML', ['fad', 'nadh', 'shg', 'intensity', 'orr'],
                           device=torch.device('cpu'), label='Metastases', mask_on=True)

    if FAST_TEST:
        ds_train.augmented = False
        ds_eval.augmented = False
        ds_train.normalize_method = 'preset'
        ds_eval.normalize_method = 'preset'
        ds_train.transforms = None
        ds_eval.transforms = None
        ds_train.to(device)
        ds_eval.to(device)
        total_epochs = 3
        batch_size = FAST_BATCH
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
        total_epochs = 500
        batch_size = BATCH

    lr = 1e-8

    subsampler = torch.utils.data.sampler.SubsetRandomSampler(range(ds_train.patient_count))
    all_idx = [i for i in subsampler if len(ds_train.get_patient_subset(i)) > 0]

    stageI_pts = []
    stageII_pts = []
    patient_names = {}
    for i in all_idx:
        name = ds_train.get_patient_name(i)
        patient_names[i] = name
        if isinstance(name, str) and name.endswith('_StageI'):
            stageI_pts.append(i)
        else:
            stageII_pts.append(i)

    print(f'Found {len(stageII_pts)} Stage II patients and {len(stageI_pts)} Stage I patients.')

    train_pts = [26, 22, 28, 24, 33, 17, 31, 25, 27, 21, 13, 16, 35, 19, 20, 15, 32]
    test_stageII_pts = [23, 18, 34, 37, 36, 14, 29, 30]
    test_stageI_pts = list(stageI_pts)

    train_idx = [im for pt in train_pts for im in ds_train.get_patient_subset(pt)]
    random.shuffle(train_idx)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds_train, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=0
    )

    model = ResNet18NPlaned(ds_train.shape, start_width=64, n_classes=1)
    if torch.cuda.is_available():
        model.to(device)

    os.makedirs('outputs', exist_ok=True)
    print('\n=== Training model on fixed Stage II training set ===')
    best_path = train_one_model(train_loader, model, device, total_epochs=total_epochs, lr=lr)
    if best_path:
        model.load_state_dict(torch.load(best_path))

    summary_rows = []

    for pool in POOL_METHODS:
        print(f'\n=== Pooling method: {pool} ===')
        base_dir = f'outputs/{model.name}/fixed_split/{pool}'
        os.makedirs(base_dir, exist_ok=True)

        # TRAIN patient-wise
        scores_train_pt, labels_train_pt, _ = patient_wise_loader_outputs(
            model, ds_train, train_pts, device, pool_method=pool,
            save_csv_path=os.path.join(base_dir, 'image_outputs_train.csv')
        )

        # ROC threshold from training
        scores_train_dict, fig_train = score_model(
            model,
            (scores_train_pt, labels_train_pt),
            print_results=False,
            make_plot=True,
            threshold_type='roc'
        )
        fig_train.savefig(os.path.join(base_dir, 'combined_plot_train_ROCopt.png'))
        plt.close(fig_train)

        thr_train = scores_train_dict.get('Optimal Threshold from ROC', 0.5)
        roc_auc_train = scores_train_dict.get('ROC-AUC', float('nan'))
        pr_auc_train = scores_train_dict.get('Average Precision', float('nan'))

        print(f'ROC-optimized threshold (training): {thr_train:.4f}')

        # TRAIN confusion + accuracy with fixed threshold
        metrics_train_fixed = evaluate_fixed_threshold(
            model, scores_train_pt, labels_train_pt, threshold=thr_train,
            out_dir=base_dir, label_tag='train_confusion'
        )

        summary_rows.append({
            'set': 'train_stageII',
            'pool_method': pool,
            'roc_auc': roc_auc_train,
            'pr_auc': pr_auc_train,
            'threshold_from_train': thr_train,
            'accuracy': metrics_train_fixed.get('Accuracy', float('nan'))
        })

        # TEST Stage II
        scores_testII_pt, labels_testII_pt, _ = patient_wise_loader_outputs(
            model, ds_eval, test_stageII_pts, device, pool_method=pool,
            save_csv_path=os.path.join(base_dir, 'image_outputs_test_stageII.csv')
        )
        metrics_testII = evaluate_fixed_threshold(
            model, scores_testII_pt, labels_testII_pt, threshold=thr_train,
            out_dir=base_dir, label_tag='test_stageII'
        )

        summary_rows.append({
            'set': 'test_stageII',
            'pool_method': pool,
            'roc_auc': metrics_testII.get('ROC-AUC', float('nan')),
            'pr_auc': metrics_testII.get('Average Precision', float('nan')),
            'threshold_from_train': thr_train,
            'accuracy': metrics_testII.get('Accuracy', float('nan'))
        })

        # TEST Stage I
        scores_testI_pt, labels_testI_pt, _ = patient_wise_loader_outputs(
            model, ds_eval, test_stageI_pts, device, pool_method=pool,
            save_csv_path=os.path.join(base_dir, 'image_outputs_test_stageI.csv')
        )
        metrics_testI = evaluate_fixed_threshold(
            model, scores_testI_pt, labels_testI_pt, threshold=thr_train,
            out_dir=base_dir, label_tag='test_stageI'
        )

        summary_rows.append({
            'set': 'test_stageI',
            'pool_method': pool,
            'roc_auc': metrics_testI.get('ROC-AUC', float('nan')),
            'pr_auc': metrics_testI.get('Average Precision', float('nan')),
            'threshold_from_train': thr_train,
            'accuracy': metrics_testI.get('Accuracy', float('nan'))
        })

    pd.DataFrame(summary_rows).to_csv('outputs/fixed_split_summary_trainThreshold_confusion.csv', index=False)
    print('\nDone. Results saved to outputs/fixed_split_summary_trainThreshold_confusion.csv')


if __name__ == '__main__':
    main()
