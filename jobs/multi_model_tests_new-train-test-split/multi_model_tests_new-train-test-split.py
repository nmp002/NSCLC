# Fast smoke-test version of multi_model_tests_with_StageI.py
# Toggle FAST_TEST to switch between smoke test and full run
FAST_TEST = False

# Import packages
import os
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms.v2 as tvt
from matplotlib import pyplot as plt
import random

from sklearn.metrics import roc_auc_score

from my_modules.models.classifier_models import *
from my_modules.scripts.model_metrics import score_model
from my_modules.scripts.helper_functions import set_seed
from my_modules.scripts.dataset import NSCLCDataset


# Pooling options: 'min', 'max', 'mean', 'median', 'bottom15', 'top15'
POOL_METHOD = 'bottom15'


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


def pool_patient_scores(outs, method=POOL_METHOD, bottom_frac=0.15):
    """
    outs: list or 1D-array of per-image model outputs (higher => more non-metastatic).
    method: pooling method.
    """
    if len(outs) == 0:
        return float('nan')
    arr = np.array(outs)
    match method.lower():
        case 'min':
            return float(np.min(arr))
        case 'max':
            return float(np.max(arr))
        case 'mean':
            return float(np.mean(arr))
        case 'median':
            return float(np.median(arr))
        case 'bottom15':
            k = max(1, int(np.ceil(len(arr) * bottom_frac)))
            s = np.sort(arr)[:k]
            return float(np.mean(s))
        case 'top15':
            k = max(1, int(np.ceil(len(arr) * bottom_frac)))
            s = np.sort(arr)[-k:]
            return float(np.mean(s))
        case _:
            # fallback to median
            return float(np.median(arr))


def patient_wise_loader_outputs(model, dataset, patient_indices, device, pool_method=POOL_METHOD):
    """Aggregate model outputs to patient level using configurable pooling rule."""
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
                out = model(x)
                out_val = out.cpu().detach().squeeze().item()
                outs.append(out_val)
            if len(outs) == 0:
                continue

            score = pool_patient_scores(outs, method=pool_method)
            patient_scores.append(score)
            patient_labels.append(dataset.get_patient_label(pt_idx).item())

            # Diagnostic prints only in FAST_TEST
            if FAST_TEST:
                print("---------------------------------")
                print("-----INDIVIDUAL IMAGE OUTPUTS-----")
                print(f"Patient Index: {pt_idx}")
                print(f"Image Outputs: {outs}")
                print(f"Patient Score ({pool_method}): {score}")
                print(f"Patient Label: {dataset.get_patient_label(pt_idx).item()}")
                print("---------------------------------")
    return torch.tensor(patient_scores), torch.tensor(patient_labels)


def main():
    # Set random seed for reproducibility
    set_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Set up multiprocessing
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

    # FAST_TEST changes: reduce heavy ops
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

    # Remove patients with no image indices
    patient_subsets = [train_data.get_patient_subset(i) for i in idx]
    idx_for_removal = []
    for i, subset in enumerate(patient_subsets):
        if len(subset) == 0:
            idx_for_removal.append(idx[i])
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

    # Get labels for Stage II patients (0 = metastatic, 1 = non-metastatic)
    labels_stageII = [train_data.get_patient_label(i).item() for i in stageII_pts]

    # Separate Stage II patients by label
    paired = list(zip(stageII_pts, labels_stageII))
    random.shuffle(paired)
    zeros = [i for i, l in paired if int(l) == 0]  # metastatic
    ones = [i for i, l in paired if int(l) == 1]   # non-metastatic

    print(f'Stage II metastatic (label=0): {len(zeros)} patients')
    print(f'Stage II non-metastatic (label=1): {len(ones)} patients')

    # New train/test split:
    # From Stage II: randomly move 3 non-met and 5 met to TEST
    if len(ones) < 3 or len(zeros) < 5:
        raise RuntimeError('Not enough Stage II patients in a class to sample requested test set.')

    # Use random.sample but deterministic because of seeded RNG above
    test_from_nonmet = random.sample(ones, 3)
    test_from_met = random.sample(zeros, 5)

    train_pts = [pt for pt in stageII_pts if pt not in (test_from_nonmet + test_from_met)]
    test_pts = stageI_pts + test_from_nonmet + test_from_met

    # Debug counts check
    train_nonmet = sum(1 for i in train_pts if int(train_data.get_patient_label(i).item()) == 1)
    train_met = sum(1 for i in train_pts if int(train_data.get_patient_label(i).item()) == 0)
    test_nonmet = sum(1 for i in test_pts if int(train_data.get_patient_label(i).item()) == 1)
    test_met = sum(1 for i in test_pts if int(train_data.get_patient_label(i).item()) == 0)

    print('\nSPLIT COUNTS (after StageII reassignment):')
    print(f'  TRAIN StageII: {len(train_pts)} patients -> {train_nonmet} non-met, {train_met} met')
    print(f'  TEST (StageI + selected StageII): {len(test_pts)} patients -> {test_nonmet} non-met, {test_met} met')

    # FAST_TEST deterministic truncation (optional)
    if FAST_TEST:
        n_per_split = 1
        train_pts = train_pts[:n_per_split]
        test_pts = test_pts[:n_per_split]
        print('FAST_TEST enabled: using first patient from each split only.')

    # Print final splits
    print('\nFINAL SPLITS (patient indices and names):')
    print(f'  TRAIN (n={len(train_pts)})')
    for i in train_pts:
        print(f'    idx {i}: {patient_names[i]} (label={train_data.get_patient_label(i).item()})')
    print(f'  TEST (n={len(test_pts)})')
    for i in test_pts:
        print(f'    idx {i}: {patient_names[i]} (label={train_data.get_patient_label(i).item()})')

    # Flatten indices for DataLoaders (image indices)
    train_idx = [train_data.get_patient_subset(i) for i in train_pts]
    train_idx = [im for i in train_idx for im in i]
    random.shuffle(train_idx)

    test_idx = [eval_test_data.get_patient_subset(i) for i in test_pts]
    test_idx = [im for i in test_idx for im in i]
    random.shuffle(test_idx)

    comb_pts = test_pts[:]  # now eval is removed, so comb == test
    comb_idx = [eval_test_data.get_patient_subset(i) for i in comb_pts]
    comb_idx = [im for i in comb_idx for im in i]
    random.shuffle(comb_idx)

    # Image count summaries
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

    # Create dataloaders
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
                                # other models omitted for speed

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
    eval_loss = [[] for _ in range(len(models))]   # kept for compatibility but unused
    eval_auc = [[] for _ in range(len(models))]    # kept for compatibility but unused
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

        # Save best by training AUC (no validation set available)
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

    # Testing (patient-wise)
    headers = ['Best Test']
    data = [[] for _ in range(len(models))]

    for i, model in enumerate(models):
        print(f'\n>>> {model.name} patient-wise testing...')
        # safe load: prefer Best, fallback to an epoch checkpoint if Best missing, else skip
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

        scores_pt, labels_pt = patient_wise_loader_outputs(model, eval_test_data, test_pts, device, pool_method=POOL_METHOD)
        print(f"Patient scores: {scores_pt}, Patient labels: {labels_pt}")
        scores, fig = score_model(model, (scores_pt, labels_pt), print_results=True, make_plot=True, threshold_type='roc')
        fig.savefig(f'outputs/{model.name}/plots/patientwise_best_eval_{model.name}_on_test.png')
        plt.close(fig)
        with open(f'outputs/{model.name}/results.txt', 'a') as f:
            f.write(f'\n>>> {model.name} patient-wise test results (pool={POOL_METHOD})...')
            for key, item in scores.items():
                if 'Confusion' not in key:
                    f.write(f'|\t{key:<35} {format_metric(item):>10}\t|\n')
            f.write('_____________________________________________________\n')
        data[i].append(scores['ROC-AUC'])

    auc_table = pd.DataFrame(data=data, index=[model.name for model in models], columns=headers)
    auc_table.to_csv(f'outputs/auc_summary.csv', index_label='Model')

    # Plot and save epoch-wise outputs (train-only)
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


# Run
if __name__ == '__main__':
    main()
