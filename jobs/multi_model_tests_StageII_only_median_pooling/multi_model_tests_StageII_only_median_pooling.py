# multi_model_tests_stageII_only_median_patientwise.py
FAST_TEST = False

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

from sklearn.metrics import roc_auc_score

from my_modules.models.classifier_models import *
from my_modules.scripts.model_metrics import score_model
from my_modules.scripts.helper_functions import set_seed
from my_modules.scripts.dataset import NSCLCDataset

# params
POOL_METHOD = 'median'   # user requested median pooling
FAST_TEST_BATCH = 8
NORMAL_BATCH = 64
TEST_THRESHOLD = 0.5     # not used for ROC-optimized workflow; kept for compatibility

def pool_patient_scores_median(outs):
    """Median pooling (outs: list of floats)."""
    if len(outs) == 0:
        return float('nan')
    return float(np.median(np.array(outs, dtype=float)))

def patient_wise_loader_outputs(model, dataset, patient_indices, device, pool_method=POOL_METHOD, save_csv=True):
    """
    Run model on images for each patient in patient_indices and pool using pool_method.
    Returns: (scores_tensor, labels_tensor, patient_order_list)
    """
    model.eval()
    patient_scores = []
    patient_labels = []
    patient_order = []
    image_outputs_log = []

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

            # pooling (only median implemented for current request)
            if pool_method == 'median':
                score = pool_patient_scores_median(outs)
            else:
                # fallback to median if other method accidentally passed
                score = pool_patient_scores_median(outs)

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
                print("-----INDIVIDUAL IMAGE OUTPUTS-----")
                print(f"Patient Index: {pt_idx}")
                print(f"Image Outputs: {outs}")
                print(f"Patient Score ({pool_method}): {score}")
                print(f"Patient Label: {int(label)}")

    # save image-level outputs for traceability
    if save_csv:
        model_name = getattr(model, 'name', 'model')
        out_dir = f"outputs/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"image_outputs_patientwise_pool-{pool_method}.csv")
        try:
            pd.DataFrame(image_outputs_log).to_csv(csv_path, index=False)
        except Exception:
            pass

    return torch.tensor(patient_scores), torch.tensor(patient_labels), patient_order

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

def save_combined_fig_and_metrics(model_name, method, scores_pt, labels_pt, patient_order, stageI_pts, out_dir_pool, scores_dict, fig):
    """
    Save combined fig (ROC/PR/CM) returned by score_model and write metrics.txt.
    Assumes score_model was called with threshold_type='roc' so ROC-optimal threshold and CM/accuracy are in scores_dict.
    """
    os.makedirs(out_dir_pool, exist_ok=True)

    # save figure
    try:
        if fig is not None:
            fig.savefig(os.path.join(out_dir_pool, 'combined_plot.png'))
            plt.close(fig)
    except Exception:
        pass

    # save pooled scores CSV
    try:
        pd.DataFrame({'patient_index': patient_order, 'pooled_score': scores_pt.numpy(), 'label': labels_pt.numpy()}).to_csv(os.path.join(out_dir_pool, f'pooled_scores_{method}.csv'), index=False)
    except Exception:
        pass

    # extract CM, accuracy info from scores_dict (score_model 'roc' returns these)
    cm = scores_dict.get('Confusion Matrix', None)
    acc = scores_dict.get('Accuracy', None)
    bal_acc = scores_dict.get('Balanced Accuracy', None)
    roc_auc = scores_dict.get('ROC-AUC', float('nan'))
    pr_auc = scores_dict.get('Average Precision', float('nan'))

    # compute stage-wise accuracy based on patient_order and pooled preds at threshold chosen by score_model
    # score_model when run with threshold_type='roc' sets 'ROC-Optimal Threshold' and applied preds inside function returning CM
    thr = scores_dict.get('ROC-Optimal Threshold', None)
    preds = None
    if thr is not None:
        preds = (scores_pt.numpy() > float(thr)).astype(int)
    labels_np = labels_pt.numpy().astype(int)
    overall_correct = int((preds == labels_np).sum()) if preds is not None else None
    overall_total = len(labels_np)

    patient_indices = [int(x) for x in patient_order]
    stageI_mask = np.array([1 if p in stageI_pts else 0 for p in patient_indices], dtype=bool)
    stageII_mask = ~stageI_mask

    def acc_counts(mask):
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return (0, 0, float('nan'))
        correct = int((preds[idxs] == labels_np[idxs]).sum()) if preds is not None else 0
        total = len(idxs)
        acc = correct / total if total > 0 else float('nan')
        return (correct, total, acc)

    s1c, s1t, s1acc = acc_counts(stageI_mask)
    s2c, s2t, s2acc = acc_counts(stageII_mask)

    # write metrics file
    metrics_path = os.path.join(out_dir_pool, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Pooling method: {method}\n")
        f.write("\n")
        f.write(f"ROC-AUC: {format_metric(roc_auc)}\n")
        f.write(f"PR-AUC (Average Precision): {format_metric(pr_auc)}\n")
        f.write("\n")
        if overall_correct is not None:
            f.write(f"Overall: {overall_correct}/{overall_total} correct -> {overall_correct/overall_total:.6f}\n")
        else:
            f.write("Overall: N/A\n")
        f.write(f"Stage I: {s1c}/{s1t} correct -> {s1acc if not math.isnan(s1acc) else 'N/A'}\n")
        f.write(f"Stage II: {s2c}/{s2t} correct -> {s2acc if not math.isnan(s2acc) else 'N/A'}\n")

def main():
    # reproducibility
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

    # prepare datasets (Stage I present but will be excluded from train/test)
    train_data = NSCLCDataset('NSCLC_Data_for_ML', ['fad','nadh','shg','intensity','orr'],
                              device=torch.device('cpu'), label='Metastases', mask_on=True)
    eval_test_data = NSCLCDataset('NSCLC_Data_for_ML', ['fad','nadh','shg','intensity','orr'],
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
                                             tvt.RandomRotation(degrees=(-180,180))])
        eval_test_data.augment()
        eval_test_data.normalize_method = 'preset'
        eval_test_data.to(device)

    # Build patient list and remove empty patients
    subsampler = torch.utils.data.sampler.SubsetRandomSampler(range(train_data.patient_count))
    idx = [i for i in subsampler]
    patient_subsets = [train_data.get_patient_subset(i) for i in idx]
    idx_for_removal = [idx[i] for i, subset in enumerate(patient_subsets) if len(subset) == 0]
    for ix in idx_for_removal:
        idx.remove(ix)

    # Split Stage I vs Stage II
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

    # Ensure Stage II present
    if len(stageII_pts) == 0:
        raise RuntimeError('No Stage II patients detected.')

    # Extract Stage II labels
    labels_stageII = [train_data.get_patient_label(i).item() for i in stageII_pts]
    paired = list(zip(stageII_pts, labels_stageII))
    random.shuffle(paired)
    zeros = [i for i,l in paired if int(l) == 0]  # metastatic
    ones = [i for i,l in paired if int(l) == 1]   # non-metastatic

    print(f'Stage II metastatic (label=0): {len(zeros)} patients')
    print(f'Stage II non-metastatic (label=1): {len(ones)} patients')

    # Sample test set: 4 non-met + 4 met
    if len(ones) < 4 or len(zeros) < 4:
        raise RuntimeError('Not enough Stage II patients in a class to sample requested test set.')

    test_from_nonmet = random.sample(ones, 4)
    test_from_met = random.sample(zeros, 4)

    train_pts = [pt for pt in stageII_pts if pt not in (test_from_nonmet + test_from_met)]
    test_pts = test_from_nonmet + test_from_met

    # Sanity counts
    train_nonmet = sum(1 for i in train_pts if int(train_data.get_patient_label(i).item()) == 1)
    train_met = sum(1 for i in train_pts if int(train_data.get_patient_label(i).item()) == 0)
    test_nonmet = sum(1 for i in test_pts if int(train_data.get_patient_label(i).item()) == 1)
    test_met = sum(1 for i in test_pts if int(train_data.get_patient_label(i).item()) == 0)

    print('\nSPLIT COUNTS:')
    print(f'  TRAIN StageII: {len(train_pts)} patients -> {train_nonmet} non-met, {train_met} met')
    print(f'  TEST StageII: {len(test_pts)} patients -> {test_nonmet} non-met, {test_met} met')

    if FAST_TEST:
        train_pts = train_pts[:1]
        test_pts = test_pts[:1]
        print('FAST_TEST: truncating to 1 pt per split.')

    # Print final splits
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

    # Dataloaders
    batch_size = FAST_TEST_BATCH if FAST_TEST else NORMAL_BATCH
    train_set = torch.utils.data.Subset(train_data, train_idx)
    test_set = torch.utils.data.Subset(eval_test_data, test_idx)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Models
    models = [ResNet18NPlaned(train_data.shape, start_width=64, n_classes=1)]
    if not FAST_TEST:
        models[len(models):] = [CNNet(train_data.shape), RegularizedCNNet(train_data.shape)]
    for m in models:
        if torch.cuda.is_available() and not next(m.parameters()).is_cuda:
            m.to(device)

    # Hyperparams
    if FAST_TEST:
        epochs = [1,5]
        total_epochs = max(epochs)
        lr = 1e-4
    else:
        epochs = [250,500,1500,2000,2500]
        total_epochs = epochs[-1]
        lr = 1e-8

    loss_function = torch.nn.BCELoss()
    optimizers = [torch.optim.Adam(m.parameters(), lr=lr, weight_decay=0.01) for m in models]

    # Output prep
    for m in models:
        os.makedirs(f'outputs/{m.name}/plots', exist_ok=True)
        os.makedirs(f'outputs/{m.name}/models', exist_ok=True)
        with open(f'outputs/{m.name}/results.txt','w') as f:
            f.write(f'{m.name} Results\n')
    if not os.path.exists('outputs/results.txt'):
        with open('outputs/results.txt','w') as f:
            f.write('Overall Results\n')

    train_loss = [[] for _ in range(len(models))]
    train_auc = [[] for _ in range(len(models))]
    best_score = [0 for _ in range(len(models))]

    # Training loop (no external validation)
    for ep in range(total_epochs):
        print(f'\nEpoch {ep+1}')
        epoch_loss = [0 for _ in range(len(models))]
        outs = [torch.tensor([]) for _ in range(len(models))]
        targets = [torch.tensor([]) for _ in range(len(models))]

        for m in models:
            m.train()
        for x, target in train_loader:
            x = x.to(device)
            target = target.to(device)
            for i, m in enumerate(models):
                out = m(x)
                outs[i] = torch.cat((outs[i], out.cpu().detach()), dim=0)
                targets[i] = torch.cat((targets[i], target.cpu().detach()), dim=0)
                loss = loss_function(out, target.unsqueeze(1))
                optimizers[i].zero_grad()
                loss.backward()
                epoch_loss[i] += loss.item()
                optimizers[i].step()

        for el, tl, ta, tx, ot, m in zip(epoch_loss, train_loss, train_auc, targets, outs, models):
            tl.append(el / (len(train_set) if len(train_set) > 0 else 1))
            try:
                ta.append(roc_auc_score(tx, ot))
            except Exception:
                ta.append(0.0)

        # Save best by training AUC (Option B requires reloading Best model later)
        for i, (m, tl, ta) in enumerate(zip(models, train_loss, train_auc)):
            train_auc_val = ta[-1] if len(ta) > 0 else 0.0
            print(f'>>> {m.name}: Train Loss {tl[-1] if len(tl)>0 else 0.0:.4f}, Train AUC {train_auc_val:.4f}')
            with open(f'outputs/{m.name}/results.txt','a') as f:
                f.write(f'\nEpoch {ep+1} >>> Train Loss {tl[-1] if len(tl)>0 else 0.0:.4f}, Train AUC {train_auc_val:.4f}\n')

            if train_auc_val > best_score[i] or FAST_TEST:
                best_score[i] = train_auc_val
                torch.save(m.state_dict(), f'outputs/{m.name}/models/Best {m.name}.pth')
                with open(f'outputs/{m.name}/results.txt','a') as f:
                    f.write(f'New best {m.name} saved at epoch {ep+1} with Train AUC {train_auc_val:.4f}\n')
                with open('outputs/results.txt','a') as f:
                    f.write(f'New best {m.name} saved at epoch {ep+1} with Train AUC {train_auc_val:.4f}\n')

            if (ep+1) in epochs:
                torch.save(m.state_dict(), f'outputs/{m.name}/models/Epochs {ep+1} {m.name}.pth')

    # After training: reload Best model and compute patient-wise train and test metrics using score_model with threshold_type='roc'
    overall_results = []
    for i, m in enumerate(models):
        print(f'\n=== Evaluating model patient-wise (median pooling) for {m.name} ===')
        best_path = f'outputs/{m.name}/models/Best {m.name}.pth'
        if os.path.exists(best_path):
            m.load_state_dict(torch.load(best_path))
        else:
            print(f'No Best model found for {m.name}, skipping evaluation.')
            continue

        # TRAIN patient-wise scores (using the saved best model)
        scores_train_pt, labels_train_pt, train_order = patient_wise_loader_outputs(m, train_data, train_pts, device, pool_method='median', save_csv=True)
        if len(scores_train_pt) > 0:
            try:
                scores_train_dict, fig_train = score_model(m, (scores_train_pt, labels_train_pt), print_results=True, make_plot=True, threshold_type='roc')
            except Exception as e:
                print(f'score_model failed for train on {m.name}: {e}')
                scores_train_dict = {}
                fig_train = None
            out_dir_train = f'outputs/{m.name}/plots/pool_median/train'
            save_combined_fig_and_metrics(m.name, 'median_train', scores_train_pt, labels_train_pt, train_order, [], out_dir_train, scores_train_dict, fig_train)
        else:
            print('No training patient scores returned.')

        # TEST patient-wise scores (using the saved best model)
        scores_test_pt, labels_test_pt, test_order = patient_wise_loader_outputs(m, eval_test_data, test_pts, device, pool_method='median', save_csv=True)
        if len(scores_test_pt) > 0:
            try:
                scores_test_dict, fig_test = score_model(m, (scores_test_pt, labels_test_pt), print_results=True, make_plot=True, threshold_type='roc')
            except Exception as e:
                print(f'score_model failed for test on {m.name}: {e}')
                scores_test_dict = {}
                fig_test = None
            out_dir_test = f'outputs/{m.name}/plots/pool_median/test'
            # For stage masks, provide stageI_pts list (empty here because Stage I not used) so Stage I/II breakdown will show Stage I = 0
            save_combined_fig_and_metrics(m.name, 'median', scores_test_pt, labels_test_pt, test_order, [], out_dir_test, scores_test_dict, fig_test)
        else:
            print('No testing patient scores returned.')

        # append summary per model
        if 'scores_test_dict' in locals() and scores_test_pt is not None:
            roc = scores_test_dict.get('ROC-AUC', float('nan'))
            pr = scores_test_dict.get('Average Precision', float('nan'))
            overall_results.append({'model': m.name, 'pool_method': 'median', 'roc_auc': roc, 'pr_auc': pr})

    # Save overall comparison
    if len(overall_results) > 0:
        pd.DataFrame(overall_results).to_csv('outputs/pooling_method_comparison_stageII_median.csv', index=False)

    # Save training-only summary CSV (loss + train_auc)
    for (m, tl, ta) in zip(models, train_loss, train_auc):
        outputs = [[a, c] for (a, c) in zip(tl, ta)]
        try:
            pd.DataFrame(data=outputs, index=range(1, total_epochs+1), columns=['Training Loss','Training ROC-AUC']).to_csv(f'outputs/{m.name}/tabular.csv', index_label='Epoch')
        except Exception:
            pass

if __name__ == '__main__':
    main()
