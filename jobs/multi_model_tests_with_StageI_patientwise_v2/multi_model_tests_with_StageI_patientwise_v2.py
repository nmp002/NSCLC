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


def patient_wise_loader_outputs(model, dataset, patient_indices, device):
    """Aggregate model outputs to patient level using max output rule."""
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
                # handle tensors of shape (1,1) or (1,)
                out_val = out.cpu().detach().squeeze().item()
                outs.append(out_val)
            if len(outs) == 0:
                continue
            max_score = max(outs)  # “most non-metastatic” output
            patient_scores.append(max_score)
            patient_labels.append(dataset.get_patient_label(pt_idx).item())
    return torch.tensor(patient_scores), torch.tensor(patient_labels)


def main():
    # Set random seed for reproducibility
    set_seed(42)
    random.seed(42)

    # Set up multiprocessing
    print(f'Num cores: {mp.cpu_count()}')
    print(f'Num GPUs: {torch.cuda.device_count()}')
    try:
        mp.set_start_method('forkserver', force=True)
    except RuntimeError:
        # start method already set in some environments
        pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ################
    # Prepare data #
    ################
    # To independent but identical (except for transformations) datasets
    train_data = NSCLCDataset('NSCLC_Data_for_ML', ['fad', 'nadh', 'shg', 'intensity', 'orr'],
                              device=torch.device('cpu'), label='Metastases', mask_on=True)
    eval_test_data = NSCLCDataset('NSCLC_Data_for_ML', ['fad', 'nadh', 'shg', 'intensity', 'orr'],
                                  device=torch.device('cpu'), label='Metastases', mask_on=True)

    # FAST_TEST changes
    if FAST_TEST:
        # reduce augmentation and heavy ops
        train_data.augmented = False
        train_data.augment_patients = False
        eval_test_data.augmented = False
        eval_test_data.augment_patients = False
        train_data.normalize_method = 'preset'
        eval_test_data.normalize_method = 'preset'
        # lighter transforms (or none)
        train_data.transforms = None
        eval_test_data.transforms = None
        # move to device
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

    # Random split datasets (start from all patients)
    subsampler = torch.utils.data.sampler.SubsetRandomSampler(range(train_data.patient_count))
    idx = [i for i in subsampler]

    # Get the image indices for all patients as nested lists
    patient_subsets = [train_data.get_patient_subset(i) for i in idx]

    # Find and remove any patients with no image indices
    idx_for_removal = []
    for i, subset in enumerate(patient_subsets):
        if len(subset) == 0:
            idx_for_removal.append(idx[i])
    for ix in idx_for_removal:
        idx.remove(ix)

    # Stage I handling
    stageI_pts = []
    stageII_pts = []
    patient_names = {}  # map idx -> name for printing and bookkeeping

    for i in idx:
        # use dataset helper to get the patient name (Slide Name or Subject)
        name = train_data.get_patient_name(i)
        patient_names[i] = name
        # Detect Stage I by suffix
        if isinstance(name, str) and name.endswith('_StageI'):
            stageI_pts.append(i)
        else:
            stageII_pts.append(i)

    # sanity prints
    print(f'Found {len(stageII_pts)} Stage II patients and {len(stageI_pts)} Stage I patients.')
    # Print the actual patient names for verification
    print('Stage I patient names (to be used as TEST set):')
    for i in stageI_pts:
        print(f'  - idx {i}: {patient_names[i]}')
    print('Stage II patient names (to be split into TRAIN / EVAL):')
    for i in stageII_pts:
        print(f'  - idx {i}: {patient_names[i]}')

    # If there are no Stage II patients, raise
    if len(stageII_pts) == 0:
        raise RuntimeError('No Stage II patients detected to split into train/eval.')

    # Get labels for Stage II patients only (0 = Metastatic, 1 = Non-metastatic)
    labels_stageII = [train_data.get_patient_label(i).item() for i in stageII_pts]
    image_counts = [0, 0]
    for i, label in zip(stageII_pts, labels_stageII):
        image_counts[int(label)] += len(train_data.get_patient_subset(i))

    # Separate by label
    paired = list(zip(stageII_pts, labels_stageII))
    random.shuffle(paired)  # shuffle the Stage II patients before stratified split
    zeros = [i for i, l in paired if int(l) == 0]  # metastatic
    ones = [i for i, l in paired if int(l) == 1]   # non-metastatic

    print(f'Stage II metastatic (label=0): {len(zeros)} patients')
    print(f'Stage II non-metastatic (label=1): {len(ones)} patients')

    # Stage II train/eval split
    total_stageII = len(stageII_pts)
    desired_train_stageII = int(round(total_stageII * 13 / 25)) if total_stageII == 25 else None
    if desired_train_stageII is None:
        desired_train_stageII = int(round(total_stageII * 13 / 25))
    desired_train = desired_train_stageII
    desired_eval = total_stageII - desired_train

    # To keep class balance, compute class-wise train counts using proportional allocation with rounding by remainder
    class_lists = [zeros, ones]
    class_train_counts = []
    remainders = []
    for cl in class_lists:
        exact = (len(cl) * desired_train) / total_stageII
        floor_count = int(np.floor(exact))
        class_train_counts.append(floor_count)
        remainders.append((exact - floor_count, cl))  # store remainder and corresponding list

    # Distribute the remaining train slots according to largest fractional remainder
    current_sum = sum(class_train_counts)
    remaining_to_assign = desired_train - current_sum
    remainders_sorted = sorted(enumerate([r[0] for r in remainders]), key=lambda x: x[1], reverse=True)
    idx_order = [r[0] for r in remainders_sorted]
    k = 0
    while remaining_to_assign > 0:
        class_train_counts[idx_order[k % len(idx_order)]] += 1
        remaining_to_assign -= 1
        k += 1

    # Now actually pick first N from each shuffled class list as train, remainder as eval
    train_pts = []
    eval_pts = []
    for cl_list, n_train in zip(class_lists, class_train_counts):
        train_from_class = cl_list[:n_train]
        eval_from_class = cl_list[n_train:]
        train_pts.extend(train_from_class)
        eval_pts.extend(eval_from_class)

    # Double-check totals
    assert len(train_pts) + len(eval_pts) == total_stageII
    assert len(train_pts) == desired_train
    assert len(eval_pts) == desired_eval

    # Test set is ALL Stage I patients
    test_pts = list(stageI_pts)

    # For FAST_TEST use fixed small subsets (deterministic)
    if FAST_TEST:
        n_per_split = 4
        train_pts = train_pts[:n_per_split]
        eval_pts = eval_pts[:n_per_split]
        test_pts = test_pts[:n_per_split]
        print('FAST_TEST enabled: using first patient from each split only.')

    # Print the chosen splits (names)
    print('\nFINAL SPLITS (patient indices and names):')
    print(f'  TRAIN (n={len(train_pts)})')
    for i in train_pts:
        print(f'    idx {i}: {patient_names[i]} (label={train_data.get_patient_label(i).item()})')
    print(f'  EVAL (n={len(eval_pts)})')
    for i in eval_pts:
        print(f'    idx {i}: {patient_names[i]} (label={train_data.get_patient_label(i).item()})')
    print(f'  TEST (Stage I) (n={len(test_pts)})')
    for i in test_pts:
        print(f'    idx {i}: {patient_names[i]} (label={train_data.get_patient_label(i).item()})')

    # Flatten indices for DataLoaders (these are image indices, not patient indices)
    train_idx = [train_data.get_patient_subset(i) for i in train_pts]
    train_idx = [im for i in train_idx for im in i]
    random.shuffle(train_idx)

    eval_idx = [eval_test_data.get_patient_subset(i) for i in eval_pts]
    eval_idx = [im for i in eval_idx for im in i]
    random.shuffle(eval_idx)

    test_idx = [eval_test_data.get_patient_subset(i) for i in test_pts]
    test_idx = [im for i in test_idx for im in i]
    random.shuffle(test_idx)

    comb_pts = eval_pts + test_pts
    comb_idx = [eval_test_data.get_patient_subset(i) for i in comb_pts]
    comb_idx = [im for i in comb_idx for im in i]
    random.shuffle(comb_idx)

    # Image count summaries
    train_image_counts = [0, 0]
    for pt in train_pts:
        label = int(train_data.get_patient_label(pt).item())
        train_image_counts[label] += len(train_data.get_patient_subset(pt))

    eval_image_counts = [0, 0]
    for pt in eval_pts:
        label = int(eval_test_data.get_patient_label(pt).item())
        eval_image_counts[label] += len(eval_test_data.get_patient_subset(pt))

    test_image_counts = [0, 0]
    for pt in test_pts:
        label = int(eval_test_data.get_patient_label(pt).item())
        test_image_counts[label] += len(eval_test_data.get_patient_subset(pt))

    comb_image_counts = [eval_image_counts[0] + test_image_counts[0], eval_image_counts[1] + test_image_counts[1]]

    print(f'\nTraining set summary: {len(train_pts)} patients, {len(train_idx)} images. Image counts per class: {train_image_counts}')
    print(f'Evaluation set summary: {len(eval_pts)} patients, {len(eval_idx)} images. Image counts per class: {eval_image_counts}')
    print(f'Test  set summary: {len(test_pts)} patients, {len(test_idx)} images. Image counts per class: {test_image_counts}')
    print(f'Combined Eval+Test summary: {len(comb_pts)} patients, {len(comb_idx)} images. Image counts per class: {comb_image_counts}')

    # Create dataloaders for fold
    batch_size = 8 if FAST_TEST else 64
    train_set = torch.utils.data.Subset(train_data, train_idx)
    eval_set = torch.utils.data.Subset(eval_test_data, eval_idx)
    test_set = torch.utils.data.Subset(eval_test_data, test_idx)
    comb_set = torch.utils.data.Subset(eval_test_data, comb_idx)

    # Create loaders
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size, shuffle=True, num_workers=0,
                                               drop_last=(True if len(train_idx) % batch_size == 1 else False))
    eval_loader = torch.utils.data.DataLoader(eval_set,
                                              batch_size=batch_size, shuffle=False, num_workers=0,
                                              drop_last=(True if len(eval_idx) % batch_size == 1 else False))
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size, shuffle=False, num_workers=0,
                                              drop_last=(True if len(test_idx) % batch_size == 1 else False))
    comb_loader = torch.utils.data.DataLoader(comb_set,
                                              batch_size=batch_size, shuffle=False, num_workers=0,
                                              drop_last=(True if len(comb_idx) % batch_size == 1 else False))

    #####################
    # Prepare model zoo #
    #####################
    # Keep a single simple model for smoke test
    models = [ResNet18NPlaned(train_data.shape, start_width=64, n_classes=1)]
    if not FAST_TEST:
        models[len(models):] = [CNNet(train_data.shape),
                                RegularizedCNNet(train_data.shape),
                                ParallelCNNet(train_data.shape),
                                RegularizedParallelCNNet(train_data.shape)]

    # Put all models on GPU if available
    for model in models:
        if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
            model.to(device)

    ###################
    # Hyperparameters #
    ###################
    if FAST_TEST:
        epochs = [1, 5]   # small checkpoints for quick smoke test
        total_epochs = max(epochs)
        learning_rate = 1e-4  # larger lr for quick convergence during smoke test
    else:
        epochs = [250, 500, 1500, 2000, 2500]
        total_epochs = epochs[-1]
        learning_rate = 1e-8

    loss_function = nn.BCELoss()
    optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01) for model in models]

    ###############
    # Output Prep #
    ###############
    # Prep results file (safe to overwrite in smoke test)
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
    eval_loss = [[] for _ in range(len(models))]
    eval_auc = [[] for _ in range(len(models))]
    best_score = [0 for _ in range(len(models))]

    # For each epoch
    for ep in range(total_epochs):
        print(f'\nEpoch {ep + 1}')

        # Train
        epoch_loss = [0 for _ in range(len(models))]

        # Preds and ground truth to calculate training AUC without having to re-run full set
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
            # avoid division by zero in smoke test if train_set is empty
            tl.append(el / (len(train_set) if len(train_set) > 0 else 1))
            try:
                ta.append(roc_auc_score(tx, ot))
            except Exception:
                ta.append(0.0)

        # Evaluation # Train
        epoch_loss = [0 for _ in range(len(models))]

        outs = [torch.tensor([]) for _ in range(len(models))]
        targets = [torch.tensor([]) for _ in range(len(models))]

        for model in models:
            model.eval()
        with torch.no_grad():
            for x, target in eval_loader:
                x = x.to(device)
                target = target.to(device)
                for i, model in enumerate(models):
                    out = model(x)
                    outs[i] = torch.cat((outs[i], out.cpu().detach()), dim=0)
                    targets[i] = torch.cat((targets[i], target.cpu().detach()), dim=0)
                    loss = loss_function(out, target.unsqueeze(1))
                    epoch_loss[i] += loss.item()
            for el, evl, ea, tx, ot, model in zip(epoch_loss, eval_loss, eval_auc, targets, outs, models):
                evl.append(el / (len(eval_set) if len(eval_set) > 0 else 1))
                try:
                    ea.append(roc_auc_score(tx, ot))
                except Exception:
                    ea.append(0.0)

        for i, (model, el, ea, tl) in enumerate(zip(models, eval_loss, eval_auc, train_loss)):
            train_auc_val = train_auc[i][-1] if len(train_auc[i]) > 0 else 0.0
            print(f'>>> {model.name}: Train - Loss: {tl[-1] if len(tl)>0 else 0.0}. AUC: {train_auc_val}.')
            print(f' --> Eval - Loss: {el[-1]:.4f}. AUC: {ea[-1]:.4f}.')

            with open(f'outputs/{model.name}/results.txt', 'a') as f:
                f.write(f'\nEpoch {ep + 1}'
                        f'>>> {model.name}: Train - Loss: {tl[-1] if len(tl)>0 else 0.0:.4f}. '
                        f'AUC: {train_auc_val:.4f}.'
                        f'--> Eval - Loss: {el[-1]:.4f}. AUC: {ea[-1]:.4f}.')

            if ea[-1] > best_score[i] or FAST_TEST == True:
                best_score[i] = ea[-1]
                torch.save(model.state_dict(), f'outputs/{model.name}/models/Best {model.name}.pth')
                with open(f'outputs/{model.name}/results.txt', 'a') as f:
                    f.write(f'\nNew best {model.name} saved at epoch {ep + 1} with ROC-AUC of {ea[-1]}')
                with open(f'outputs/results.txt', 'a') as f:
                    f.write(f'\nNew best {model.name} saved at epoch {ep + 1} with ROC-AUC of {ea[-1]}')

            if (ep + 1) in epochs:
                torch.save(model.state_dict(), f'outputs/{model.name}/models/Epochs {ep + 1} {model.name}.pth')

    with open(f'outputs/results.txt', 'a') as f:
        f.write(f'\n\nFinal ROC-AUC Results\n')
        for model, bs, ta, ea in zip(models, best_score, train_auc, eval_auc):
            f.write(
                f'{model.name}: Best eval AUC - {bs:.4f}. '
                f'Final Train AUC - {ta[-1] if len(ta)>0 else 0.0:.4f}. Final Eval AUC - {ea[-1] if len(ea)>0 else 0.0:.4f}\n')

    # Testing (patient-wise)
    headers = ['Best Test', 'Best Eval & Test']
    data = [[] for _ in range(len(models))]

    for i, model in enumerate(models):
        # best eval model - patient-wise test
        print(f'\n>>> {model.name} at best evaluated on patient-wise test set...')
        model.load_state_dict(torch.load(f'outputs/{model.name}/models/Best {model.name}.pth'))
        scores_pt, labels_pt = patient_wise_loader_outputs(model, eval_test_data, test_pts, device)
        print(f"Patient scores: {scores_pt}, Patient labels: {labels_pt}")
        scores, fig = score_model(model, (scores_pt, labels_pt), print_results=True, make_plot=True,
                                  threshold_type='roc')
        fig.savefig(f'outputs/{model.name}/plots/patientwise_best_eval_{model.name}_on_test.png')
        plt.close(fig)
        with open(f'outputs/{model.name}/results.txt', 'a') as f:
            f.write(f'\n>>> {model.name} patient-wise best-eval test results...')
            for key, item in scores.items():
                if 'Confusion' not in key:
                    f.write(f'|\t{key:<35} {format_metric(item):>10}\t|\n')
            f.write('_____________________________________________________\n')
        data[i].append(scores['ROC-AUC'])

        # best eval model - patient-wise eval+test
        print(f'\n>>> {model.name} at best evaluated on patient-wise eval and test sets...')
        scores_pt, labels_pt = patient_wise_loader_outputs(model, eval_test_data, comb_pts, device)
        scores, fig = score_model(model, (scores_pt, labels_pt), print_results=True, make_plot=True,
                                  threshold_type='roc')
        fig.savefig(f'outputs/{model.name}/plots/patientwise_best_eval_{model.name}_on_eval-test.png')
        plt.close(fig)
        with open(f'outputs/{model.name}/results.txt', 'a') as f:
            f.write(f'\n>>> {model.name} patient-wise best-eval eval+test results...')
            for key, item in scores.items():
                if 'Confusion' not in key:
                    f.write(f'|\t{key:<35} {format_metric(item):>10}\t|\n')
            f.write('_____________________________________________________\n')
        data[i].append(scores['ROC-AUC'])

        # checkpoint epochs - patient-wise test and eval+test
        for ep in epochs:
            headers.append(f'{ep} Epoch Test') if f'{ep} Epoch Test' not in headers else None
            print(f'\n>>> {model.name} at {ep} epochs on patient-wise test set...')
            # load epoch checkpoint if exists, otherwise skip
            ep_path = f'outputs/{model.name}/models/Epochs {ep} {model.name}.pth'
            if os.path.exists(ep_path):
                model.load_state_dict(torch.load(ep_path))
            scores_pt, labels_pt = patient_wise_loader_outputs(model, eval_test_data, test_pts, device)
            scores, fig = score_model(model, (scores_pt, labels_pt), print_results=True, make_plot=True,
                                      threshold_type='roc')
            fig.savefig(f'outputs/{model.name}/plots/patientwise_{ep}_{model.name}_test.png')
            plt.close(fig)
            with open(f'outputs/{model.name}/results.txt', 'a') as f:
                f.write(f'\n>>> {model.name} at {ep} epochs patient-wise test results...')
                for key, item in scores.items():
                    if 'Confusion' not in key:
                        f.write(f'|\t{key:<35} {format_metric(item):>10}\t|\n')
                f.write('_____________________________________________________\n')
            data[i].append(scores['ROC-AUC'])

            headers.append(f'{ep} Epoch Eval & Test') if f'{ep} Epoch Eval & Test' not in headers else None
            print(f'\n>>> {model.name} at {ep} epochs on patient-wise eval and test sets...')
            scores_pt, labels_pt = patient_wise_loader_outputs(model, eval_test_data, comb_pts, device)
            scores, fig = score_model(model, (scores_pt, labels_pt), print_results=True, make_plot=True,
                                      threshold_type='roc')
            fig.savefig(f'outputs/{model.name}/plots/patientwise_{ep}_{model.name}_eval-test.png')
            plt.close(fig)
            with open(f'outputs/{model.name}/results.txt', 'a') as f:
                f.write(f'\n>>> {model.name} at {ep} epochs patient-wise eval+test results...')
                for key, item in scores.items():
                    if 'Confusion' not in key:
                        f.write(f'|\t{key:<35} {format_metric(item):>10}\t|\n')
                f.write('_____________________________________________________\n')
            data[i].append(scores['ROC-AUC'])

    auc_table = pd.DataFrame(data=data, index=[model.name for model in models], columns=headers)
    auc_table.to_csv(f'outputs/auc_summary.csv', index_label='Model')

    # Plot and save epoch-wise outputs
    headers = ['Training Loss (average per sample)', 'Evaluation Loss (average per sample)',
               'Training ROC-AUC', 'Evaluation ROC-AUC']
    for (model, tl, el, ta, ea) in zip(models, train_loss, eval_loss, train_auc, eval_auc):
        outputs = [[a, b, c, d] for (a, b, c, d) in zip(tl, el, ta, ea)]
        output_table = pd.DataFrame(data=outputs, index=range(1, total_epochs + 1), columns=headers)
        output_table.to_csv(f'outputs/{model.name}/tabular.csv', index_label='Epoch')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        plt.suptitle(model.name)

        ax1.plot(range(1, total_epochs + 1), tl, label=f'{model.name} Training')
        ax1.plot(range(1, total_epochs + 1), el, label=f'{model.name} Evaluation')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Evaluation Losses')
        ax1.legend()

        ax2.plot(range(1, total_epochs + 1), ta, label=f'{model.name} Training')
        ax2.plot(range(1, total_epochs + 1), ea, label=f'{model.name} Evaluation')
        ax1.set_ylabel('Epochs')
        ax2.set_ylabel('AUC')
        ax2.set_title('Training and Evaluation ROC-AUC')
        ax2.legend()

        fig.savefig(f'outputs/{model.name}/plots/losses_and_aucs.png')
        plt.close(fig)


# Run
if __name__ == '__main__':
    main()
