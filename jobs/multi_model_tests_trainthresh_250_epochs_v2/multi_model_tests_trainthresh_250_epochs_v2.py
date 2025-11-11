# stageII_fixed_split_stageI_test_trainTHR.py
# Train on fixed 17 Stage II patients, test on 8 Stage II + 13 Stage I.
# Threshold is ROC-optimized on Stage II *training* set.
# Runs both median and min pooling. Produces loss + AUC curves.

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms.v2 as tvt
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
import torch.nn as nn

from my_modules.models.classifier_models import *
from my_modules.scripts.model_metrics import score_model
from my_modules.scripts.helper_functions import set_seed
from my_modules.scripts.dataset import NSCLCDataset

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
FAST_TEST = False
TOTAL_EPOCHS = 250
POOL_METHODS = ['median', 'min']

TRAIN_PTS = [26, 22, 28, 24, 33, 17, 31, 25, 27, 21, 13, 16, 35, 19, 20, 15, 32]
TEST_PTS_STAGEII = [23, 18, 34, 37, 36, 14, 29, 30]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
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


def pool_patient_scores(outs, method='median'):
    arr = np.asarray(outs, dtype=float)
    if arr.size == 0:
        return float('nan')
    m = method.lower()
    if m == 'min':
        return float(np.min(arr))
    if m == 'median':
        return float(np.median(arr))
    return float(np.median(arr))


def patient_wise_loader_outputs(model, dataset, patient_indices, device, pool_method='median'):
    model.eval()
    patient_scores = []
    patient_labels = []
    with torch.no_grad():
        for pt_idx in patient_indices:
            img_indices = dataset.get_patient_subset(pt_idx)
            outs = []
            for im_idx in img_indices:
                x, _ = dataset[im_idx]
                x = x.unsqueeze(0).to(device)
                out = model(x).cpu().detach().squeeze().item()
                outs.append(out)
            if len(outs) == 0:
                continue
            pooled = pool_patient_scores(outs, method=pool_method)
            patient_scores.append(pooled)
            patient_labels.append(dataset.get_patient_label(pt_idx).item())
    return torch.tensor(patient_scores), torch.tensor(patient_labels)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_data = NSCLCDataset('NSCLC_Data_for_ML',
                              ['fad', 'nadh', 'shg', 'intensity', 'orr'],
                              device=torch.device('cpu'),
                              label='Metastases',
                              mask_on=True)
    eval_data = NSCLCDataset('NSCLC_Data_for_ML',
                             ['fad', 'nadh', 'shg', 'intensity', 'orr'],
                             device=torch.device('cpu'),
                             label='Metastases',
                             mask_on=True)

    if FAST_TEST:
        train_data.augmented = False
        eval_data.augmented = False
        train_data.normalize_method = 'preset'
        eval_data.normalize_method = 'preset'
        train_data.transforms = None
        eval_data.transforms = None
        train_data.to(device)
        eval_data.to(device)
        total_epochs = 3
        batch_size = 8
    else:
        train_data.augment()
        train_data.normalize_method = 'preset'
        train_data.to(device)
        train_data.transforms = tvt.Compose([
            tvt.RandomVerticalFlip(p=0.25),
            tvt.RandomHorizontalFlip(p=0.25),
            tvt.RandomRotation(degrees=(-180, 180))
        ])
        eval_data.augment()
        eval_data.normalize_method = 'preset'
        eval_data.to(device)
        total_epochs = TOTAL_EPOCHS
        batch_size = 64

    # Identify Stage I patients for later testing
    subsampler = torch.utils.data.sampler.SubsetRandomSampler(range(train_data.patient_count))
    all_pt_idx = [i for i in subsampler if len(train_data.get_patient_subset(i)) > 0]
    stageI_pts = []
    patient_names = {}
    for i in all_pt_idx:
        name = train_data.get_patient_name(i)
        patient_names[i] = name
        if isinstance(name, str) and name.endswith('_StageI'):
            stageI_pts.append(i)

    print(f'Found {len(stageI_pts)} Stage I patients.')

    # Flatten image indices for training/testing
    train_img_idx = [train_data.get_patient_subset(i) for i in TRAIN_PTS]
    train_img_idx = [im for sub in train_img_idx for im in sub]
    random.shuffle(train_img_idx)
    test_img_idx_stageII = [eval_data.get_patient_subset(i) for i in TEST_PTS_STAGEII]
    test_img_idx_stageII = [im for sub in test_img_idx_stageII for im in sub]
    random.shuffle(test_img_idx_stageII)

    # Dataloaders
    train_set = torch.utils.data.Subset(train_data, train_img_idx)
    test_set_stageII = torch.utils.data.Subset(eval_data, test_img_idx_stageII)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader_stageII = torch.utils.data.DataLoader(test_set_stageII, batch_size=batch_size, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    models = [ResNet18NPlaned(train_data.shape, start_width=64, n_classes=1)]
    if not FAST_TEST:
        models += [CNNet(train_data.shape), RegularizedCNNet(train_data.shape)]
    for m in models:
        if torch.cuda.is_available() and not next(m.parameters()).is_cuda:
            m.to(device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    loss_fn = nn.BCELoss()
    optimizers = [torch.optim.Adam(m.parameters(), lr=1e-8, weight_decay=0.01) for m in models]

    os.makedirs('outputs', exist_ok=True)
    for m in models:
        os.makedirs(f'outputs/{m.name}/plots', exist_ok=True)
        os.makedirs(f'outputs/{m.name}/models', exist_ok=True)
        with open(f'outputs/{m.name}/results.txt', 'w') as f:
            f.write(f'{m.name} Results\n')

    train_loss = [[] for _ in models]
    train_auc = [[] for _ in models]
    best_score = [0.0 for _ in models]

    for ep in range(total_epochs):
        print(f'\nEpoch {ep + 1}/{total_epochs}')
        epoch_loss = [0.0 for _ in models]
        outs = [torch.tensor([]) for _ in models]
        targets = [torch.tensor([]) for _ in models]

        for m in models:
            m.train()

        for x, target in train_loader:
            x, target = x.to(device), target.to(device)
            for i, m in enumerate(models):
                out = m(x)
                outs[i] = torch.cat((outs[i], out.cpu().detach()), dim=0)
                targets[i] = torch.cat((targets[i], target.cpu().detach()), dim=0)
                loss = loss_fn(out, target.unsqueeze(1))
                optimizers[i].zero_grad()
                loss.backward()
                epoch_loss[i] += loss.item()
                optimizers[i].step()

        for i, (el, tl, ta, tx, ot, m) in enumerate(zip(epoch_loss, train_loss, train_auc, targets, outs, models)):
            tl.append(el / max(len(train_set), 1))
            try:
                ta.append(roc_auc_score(tx, ot))
            except Exception:
                ta.append(0.0)
            print(f'>>> {m.name}: Train Loss={tl[-1]:.4f}, Train AUC={ta[-1]:.4f}')
            if ta[-1] > best_score[i]:
                best_score[i] = ta[-1]
                torch.save(m.state_dict(), f'outputs/{m.name}/models/Best {m.name}.pth')

    # ------------------------------------------------------------------
    # Plot training curves
    # ------------------------------------------------------------------
    for (m, tl, ta) in zip(models, train_loss, train_auc):
        df = pd.DataFrame({'Training Loss': tl, 'Training ROC-AUC': ta}, index=range(1, total_epochs + 1))
        df.to_csv(f'outputs/{m.name}/tabular_train.csv', index_label='Epoch')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        plt.suptitle(m.name)
        ax1.plot(df.index, df['Training Loss']); ax1.set_title('Training Loss')
        ax2.plot(df.index, df['Training ROC-AUC']); ax2.set_title('Training ROC-AUC')
        fig.savefig(f'outputs/{m.name}/plots/loss_auc_curves.png'); plt.close(fig)

    # ------------------------------------------------------------------
    # Evaluation using threshold from TRAINING set
    # ------------------------------------------------------------------
    for pool_method in POOL_METHODS:
        print(f'\n=== Pooling method: {pool_method} ===')
        for m in models:
            print(f'\n>>> Evaluating model {m.name} with pooling={pool_method}')
            m.load_state_dict(torch.load(f'outputs/{m.name}/models/Best {m.name}.pth'))
            m.eval()
            out_dir = f'outputs/{m.name}/pool_{pool_method}'
            os.makedirs(out_dir, exist_ok=True)

            # --- TRAIN ROC/PR and derive threshold ---
            scores_train_pt, labels_train_pt = patient_wise_loader_outputs(m, train_data, TRAIN_PTS, device, pool_method)
            scores_train, fig_train = score_model(m, (scores_train_pt, labels_train_pt),
                                                  print_results=False, make_plot=True, threshold_type='roc')
            fig_train.savefig(os.path.join(out_dir, 'train_combined.png')); plt.close(fig_train)
            thr_train = scores_train.get('Optimal Threshold from ROC', 0.5)
            print(f"[INFO] {m.name} threshold from TRAIN set = {thr_train:.4f}")

            # --- Stage II TEST using fixed training threshold ---
            scores_testII_pt, labels_testII_pt = patient_wise_loader_outputs(m, eval_data, TEST_PTS_STAGEII, device, pool_method)
            scores_testII, fig_testII = score_model(m, (scores_testII_pt, labels_testII_pt),
                                                    print_results=False, make_plot=True,
                                                    threshold_type='fixed', threshold=float(thr_train))
            fig_testII.savefig(os.path.join(out_dir, 'test_stageII_combined.png')); plt.close(fig_testII)

            # --- Stage I TEST using same threshold ---
            stageI_pts = [i for i in patient_names if isinstance(patient_names[i], str) and patient_names[i].endswith('_StageI')]
            scores_testI_pt, labels_testI_pt = patient_wise_loader_outputs(m, eval_data, stageI_pts, device, pool_method)
            scores_testI, fig_testI = score_model(m, (scores_testI_pt, labels_testI_pt),
                                                  print_results=False, make_plot=True,
                                                  threshold_type='fixed', threshold=float(thr_train))
            fig_testI.savefig(os.path.join(out_dir, 'test_stageI_combined.png')); plt.close(fig_testI)

            with open(f'outputs/{m.name}/results.txt', 'a') as f:
                f.write(f'\n[TRAIN] {m.name} pooling={pool_method}\n')
                f.write(f'Training ROC-opt threshold = {format_metric(thr_train)}\n')
                for key, item in scores_train.items():
                    if 'Confusion' not in key:
                        f.write(f'|\t{key:<35} {format_metric(item):>10}\t|\n')
                f.write('_____________________________________________________\n')

                f.write(f'\n[TEST Stage II] {m.name} pooling={pool_method} (threshold from TRAIN)\n')
                for key, item in scores_testII.items():
                    if 'Confusion' not in key:
                        f.write(f'|\t{key:<35} {format_metric(item):>10}\t|\n')
                f.write('_____________________________________________________\n')

                f.write(f'\n[TEST Stage I] {m.name} pooling={pool_method} (threshold from TRAIN)\n')
                for key, item in scores_testI.items():
                    if 'Confusion' not in key:
                        f.write(f'|\t{key:<35} {format_metric(item):>10}\t|\n')
                f.write('_____________________________________________________\n')


if __name__ == '__main__':
    main()
