# multi_model_tests_with_StageI_use_score_model_fixed_threshold.py

FAST_TEST = False

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms.v2 as tvt
from matplotlib import pyplot as plt
import random

from sklearn.metrics import roc_auc_score

from my_modules.models.classifier_models import *
from my_modules.scripts.helper_functions import set_seed
from my_modules.scripts.dataset import NSCLCDataset
from my_modules.scripts.model_metrics import score_model

# -------------------------
# main script starts here
# -------------------------
# pooling and params
POOL_METHODS = ['min', 'max', 'mean', 'median', 'bottom15', 'top15', 'softmin', 'majority']
BOTTOM_FRAC = 0.15
SOFTMIN_TEMP = 10.0
MAJORITY_THRESHOLD = 0.5
TEST_THRESHOLD = 0.5  # fixed

def pool_patient_scores(outs, method='bottom15', bottom_frac=BOTTOM_FRAC, softmin_temp=SOFTMIN_TEMP, maj_thresh=MAJORITY_THRESHOLD):
    if len(outs) == 0:
        return float('nan')
    arr = np.array(outs, dtype=float)
    m = method.lower()
    if m == 'min':
        return float(np.min(arr))
    if m == 'max':
        return float(np.max(arr))
    if m == 'mean':
        return float(np.mean(arr))
    if m == 'median':
        return float(np.median(arr))
    if m == 'bottom15':
        k = max(1, int(np.ceil(len(arr) * bottom_frac)))
        s = np.sort(arr)[:k]
        return float(np.mean(s))
    if m == 'top15':
        k = max(1, int(np.ceil(len(arr) * bottom_frac)))
        s = np.sort(arr)[-k:]
        return float(np.mean(s))
    if m == 'softmin':
        exps = np.exp(-softmin_temp * arr)
        denom = np.sum(exps) + 1e-12
        weights = exps / denom
        return float(np.sum(weights * arr))
    if m == 'majority':
        preds = (arr > maj_thresh).astype(int)
        frac = preds.sum() / len(preds)
        return float(frac)
    return float(np.median(arr))

def patient_wise_loader_outputs(model, dataset, patient_indices, device, pool_method='bottom15', save_csv=True):
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
                print("-----INDIVIDUAL IMAGE OUTPUTS-----")
                print(f"Patient Index: {pt_idx}")
                print(f"Image Outputs: {outs}")
                print(f"Patient Score ({pool_method}): {score}")
                print(f"Patient Label: {int(label)}")
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

def main():
    set_seed(42)
    random.seed(42)
    np.random.seed(42)

    print(f'Num cores: {mp.cpu_count()}')
    print(f'Num GPUs: {torch.cuda.device_count()}')
    try:
        mp.set_start_method('forkserver', force=True)
    except Exception:
        pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare datasets
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

    subsampler = torch.utils.data.sampler.SubsetRandomSampler(range(train_data.patient_count))
    idx = [i for i in subsampler]

    patient_subsets = [train_data.get_patient_subset(i) for i in idx]
    idx_for_removal = [idx[i] for i, subset in enumerate(patient_subsets) if len(subset)==0]
    for ix in idx_for_removal:
        idx.remove(ix)

    # Partition Stage I vs Stage II
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
    if len(stageII_pts) == 0:
        raise RuntimeError('No Stage II patients detected.')

    labels_stageII = [train_data.get_patient_label(i).item() for i in stageII_pts]
    paired = list(zip(stageII_pts, labels_stageII))
    random.shuffle(paired)
    zeros = [i for i,l in paired if int(l)==0]
    ones = [i for i,l in paired if int(l)==1]

    if len(ones) < 3 or len(zeros) < 5:
        raise RuntimeError('Not enough Stage II patients in class for requested test set sampling.')

    test_from_nonmet = random.sample(ones, 3)
    test_from_met = random.sample(zeros, 5)
    train_pts = [pt for pt in stageII_pts if pt not in (test_from_nonmet + test_from_met)]
    test_pts = stageI_pts + test_from_nonmet + test_from_met

    if FAST_TEST:
        train_pts = train_pts[:1]
        test_pts = test_pts[:1]

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

    # Dataloaders
    batch_size = 8 if FAST_TEST else 64
    train_set = torch.utils.data.Subset(train_data, train_idx)
    test_set = torch.utils.data.Subset(eval_test_data, test_idx)
    comb_set = torch.utils.data.Subset(eval_test_data, comb_idx)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    comb_loader = torch.utils.data.DataLoader(comb_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Models
    models = [ResNet18NPlaned(train_data.shape, start_width=64, n_classes=1)]
    if not FAST_TEST:
        models[len(models):] = [CNNet(train_data.shape), RegularizedCNNet(train_data.shape)]
    for model in models:
        if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
            model.to(device)

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

    # Training loop (no validation)
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
            tl.append(el / (len(train_set) if len(train_set)>0 else 1))
            try:
                ta.append(roc_auc_score(tx, ot))
            except Exception:
                ta.append(0.0)

        for i, (m, tl, ta) in enumerate(zip(models, train_loss, train_auc)):
            train_auc_val = ta[-1] if len(ta)>0 else 0.0
            print(f'>>> {m.name}: Train Loss {tl[-1] if len(tl)>0 else 0.0:.4f}, AUC {train_auc_val:.4f}')
            with open(f'outputs/{m.name}/results.txt','a') as f:
                f.write(f'\nEpoch {ep+1} >>> Train Loss {tl[-1] if len(tl)>0 else 0.0:.4f}, AUC {train_auc_val:.4f}\n')

            if train_auc_val > best_score[i] or FAST_TEST:
                best_score[i] = train_auc_val
                torch.save(m.state_dict(), f'outputs/{m.name}/models/Best {m.name}.pth')
                with open(f'outputs/{m.name}/results.txt','a') as f:
                    f.write(f'New best {m.name} saved at epoch {ep+1} with AUC {train_auc_val:.4f}\n')
                with open('outputs/results.txt','a') as f:
                    f.write(f'New best {m.name} saved at epoch {ep+1} with AUC {train_auc_val:.4f}\n')

            if (ep+1) in epochs:
                torch.save(m.state_dict(), f'outputs/{m.name}/models/Epochs {ep+1} {m.name}.pth')

    # Final train summary
    with open('outputs/results.txt','a') as f:
        f.write('\nFinal AUC Results\n')
        for m, bs, ta in zip(models, best_score, train_auc):
            f.write(f'{m.name}: Best train AUC {bs:.4f}, Final Train AUC {ta[-1] if len(ta)>0 else 0.0:.4f}\n')

    # Testing: use score_model with fixed threshold to produce consistent 3-panel figs
    overall_results = []
    for i, m in enumerate(models):
        print(f'\nTesting model {m.name}...')
        best_path = f'outputs/{m.name}/models/Best {m.name}.pth'
        fallback = f'outputs/{m.name}/models/Epochs {epochs[0]} {m.name}.pth' if len(epochs)>0 else None
        if os.path.exists(best_path):
            m.load_state_dict(torch.load(best_path))
        elif fallback and os.path.exists(fallback):
            m.load_state_dict(torch.load(fallback))
        else:
            print(f'No saved model for {m.name}, skipping.')
            continue

        per_model_results = []
        for method in POOL_METHODS:
            scores_pt, labels_pt, patient_order = patient_wise_loader_outputs(m, eval_test_data, test_pts, device, pool_method=method, save_csv=True)
            if len(scores_pt)==0:
                print(f'No scores for pool method {method}, skipping.')
                continue

            # call score_model with fixed threshold
            try:
                scores_dict, fig = score_model(m, (scores_pt, labels_pt), print_results=False, make_plot=True, threshold_type='fixed', threshold=TEST_THRESHOLD)
            except Exception as e:
                print(f'score_model failed for {m.name} pool {method}: {e}')
                # fallback: compute ROC-AUC and make a simple figure
                scores_dict = {}
                try:
                    scores_dict['ROC-AUC'] = float(roc_auc_score(labels_pt.numpy(), scores_pt.numpy()))
                except Exception:
                    scores_dict['ROC-AUC'] = float('nan')
                fig = None

            # Save combined 3-panel figure using A2 folder structure inside plots/
            out_dir_pool = f'outputs/{m.name}/plots/pool_{method}'
            os.makedirs(out_dir_pool, exist_ok=True)
            fig_path = os.path.join(out_dir_pool, 'combined_plot.png')
            try:
                if fig is not None:
                    fig.savefig(fig_path)
                    plt.close(fig)
            except Exception:
                pass

            # Write metrics file (exclude threshold line per your earlier choice)
            metrics_path = os.path.join(out_dir_pool, 'metrics.txt')
            with open(metrics_path, 'w') as f:
                f.write(f"Model: {m.name}\n")
                f.write(f"Pooling method: {method}\n\n")
                f.write(f"ROC-AUC: {format_metric(scores_dict.get('ROC-AUC', float('nan')))}\n")
                f.write(f"Average Precision (PR-AUC): {format_metric(scores_dict.get('Average Precision', float('nan')))}\n")
                # confusion matrix and accuracy keys in scores_dict (if present)
                cm = scores_dict.get('Confusion Matrix', None)
                acc = scores_dict.get('Accuracy', None)
                bal_acc = scores_dict.get('Balanced Accuracy', None)
                if acc is not None:
                    f.write(f"Accuracy (@{TEST_THRESHOLD}): {format_metric(acc)}\n")
                if bal_acc is not None:
                    f.write(f"Balanced Accuracy (@{TEST_THRESHOLD}): {format_metric(bal_acc)}\n")
                if cm is not None:
                    f.write(f"Confusion Matrix (rows=true, cols=pred): {cm}\n")

            # Save pooled scores CSV
            pooled_csv = os.path.join(out_dir_pool, f'pooled_scores_{method}.csv')
            try:
                pd.DataFrame({'patient_index': patient_order, 'pooled_score': scores_pt.numpy(), 'label': labels_pt.numpy()}).to_csv(pooled_csv, index=False)
            except Exception:
                pass

            # Compute accuracy breakdown by stage using patient_order
            scores_np = scores_pt.numpy()
            labels_np = labels_pt.numpy().astype(int)
            preds = (scores_np > TEST_THRESHOLD).astype(int)
            overall_correct = int((preds == labels_np).sum())
            overall_total = len(labels_np)
            overall_acc = overall_correct / overall_total if overall_total>0 else float('nan')
            # stage masks
            patient_indices = [int(x) for x in patient_order]
            stageI_mask = np.array([1 if p in stageI_pts else 0 for p in patient_indices], dtype=bool)
            stageII_mask = ~stageI_mask

            def acc_counts(mask):
                idxs = np.where(mask)[0]
                if len(idxs)==0:
                    return (0,0,float('nan'))
                correct = int((preds[idxs] == labels_np[idxs]).sum())
                total = len(idxs)
                return (correct, total, correct/total)

            s1c, s1t, s1acc = acc_counts(stageI_mask)
            s2c, s2t, s2acc = acc_counts(stageII_mask)

            # append to per-model result table
            per_model_results.append({
                'model': m.name,
                'pool_method': method,
                'roc_auc': float(scores_dict.get('ROC-AUC', float('nan'))),
                'pr_auc': float(scores_dict.get('Average Precision', float('nan'))),
                'overall_correct': int(overall_correct),
                'overall_total': int(overall_total),
                'overall_acc': float(overall_acc),
                'stageI_correct': int(s1c),
                'stageI_total': int(s1t),
                'stageI_acc': float(s1acc) if not math.isnan(s1acc) else float('nan'),
                'stageII_correct': int(s2c),
                'stageII_total': int(s2t),
                'stageII_acc': float(s2acc) if not math.isnan(s2acc) else float('nan')
            })

            # log into model results.txt (exclude threshold value per earlier choice)
            with open(f'outputs/{m.name}/results.txt','a') as rf:
                rf.write(f'\nPOOL METHOD: {method}\n')
                rf.write(f'  ROC-AUC: {format_metric(scores_dict.get("ROC-AUC", float("nan")))}\n')
                rf.write(f'  PR-AUC: {format_metric(scores_dict.get("Average Precision", float("nan")))}\n')
                rf.write(f'  Overall: {overall_correct}/{overall_total} -> {overall_acc:.4f}\n')
                rf.write(f'  Stage I: {s1c}/{s1t} -> {s1acc if not math.isnan(s1acc) else "N/A"}\n')
                rf.write(f'  Stage II: {s2c}/{s2t} -> {s2acc if not math.isnan(s2acc) else "N/A"}\n')
                rf.write('----------------------------------------------------\n')

        # Save per-model CSV
        if len(per_model_results)>0:
            pd.DataFrame(per_model_results).to_csv(f'outputs/{m.name}/pooling_method_summary_fixed_threshold.csv', index=False)
            overall_results.extend(per_model_results)
            # concise comparison
            try:
                pd.DataFrame(per_model_results)[['pool_method','overall_acc','stageI_acc','stageII_acc','roc_auc','pr_auc']].to_csv(f'outputs/{m.name}/plots/pooling_accuracy_comparison.csv', index=False)
            except Exception:
                pass

            # print table
            print(f'\nSummary for {m.name}')
            try:
                print(pd.DataFrame(per_model_results)[['pool_method','overall_acc','stageI_acc','stageII_acc','roc_auc','pr_auc']].to_string(index=False))
            except Exception:
                pass

    # save overall comparisons
    if len(overall_results)>0:
        all_df = pd.DataFrame(overall_results)
        all_df.to_csv('outputs/pooling_method_comparison_all_models_fixed_threshold.csv', index=False)
        try:
            human_cols = ['model','pool_method','roc_auc','pr_auc','overall_correct','overall_total','overall_acc',
                          'stageI_correct','stageI_total','stageI_acc','stageII_correct','stageII_total','stageII_acc']
            all_df[human_cols].to_csv('outputs/pooling_method_accuracy_readout.csv', index=False)
        except Exception:
            all_df.to_csv('outputs/pooling_method_accuracy_readout.csv', index=False)

    # save train-only summaries
    for (m, tl, ta) in zip(models, train_loss, train_auc):
        outputs = [[a, c] for (a, c) in zip(tl, ta)]
        try:
            pd.DataFrame(data=outputs, index=range(1, total_epochs+1), columns=['Training Loss','Training ROC-AUC']).to_csv(f'outputs/{m.name}/tabular.csv', index_label='Epoch')
        except Exception:
            pass

if __name__ == '__main__':
    main()
