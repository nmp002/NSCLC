# --- Existing imports and setup remain unchanged ---
# Import packages
import os

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


def main():
    # Set random seed for reproducibility
    set_seed(42)
    random.seed(42)

    # Set up multiprocessing
    print(f'Num cores: {mp.cpu_count()}')
    print(f'Num GPUs: {torch.cuda.device_count()}')
    mp.set_start_method('forkserver', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ################
    # Prepare data #
    ################
    # To independent but identical (except for transformations) datasets
    train_data = NSCLCDataset('NSCLC_Data_for_ML', ['fad', 'nadh', 'shg', 'intensity', 'orr'],
                              device=torch.device('cpu'), label='Metastases', mask_on=True)
    train_data.augment()
    train_data.normalize_method = 'preset'
    train_data.to(device)
    train_data.transforms = tvt.Compose([tvt.RandomVerticalFlip(p=0.25),
                                         tvt.RandomHorizontalFlip(p=0.25),
                                         tvt.RandomRotation(degrees=(-180, 180))])

    eval_test_data = NSCLCDataset('NSCLC_Data_for_ML', ['fad', 'nadh', 'shg', 'intensity', 'orr'],
                                  device=torch.device('cpu'), label='Metastases', mask_on=True)
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

    # --- NEW: Stage I handling (Method 2: name endswith '_StageI' -> Stage I) ---
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

    # Get labels for Stage II patients only (0 = Metastatic, 1 = Non-metastatic per dataset definition)
    labels_stageII = [train_data.get_patient_label(i).item() for i in stageII_pts]
    image_counts = [0, 0]
    for i, label in zip(stageII_pts, labels_stageII):
        image_counts[int(label)] += len(train_data.get_patient_subset(i))

    # Separate by label (still keep them shuffled deterministically)
    paired = list(zip(stageII_pts, labels_stageII))
    random.shuffle(paired)  # shuffle the Stage II patients before stratified split
    # split back into class-specific lists (preserve the shuffle order within each class)
    zeros = [i for i, l in paired if int(l) == 0]  # metastatic
    ones = [i for i, l in paired if int(l) == 1]  # non-metastatic

    print(f'Stage II metastatic (label=0): {len(zeros)} patients')
    print(f'Stage II non-metastatic (label=1): {len(ones)} patients')

    # Desired Stage II train/eval sizes
    total_stageII = len(stageII_pts)
    desired_train_stageII = int(round(total_stageII * 13 / 25)) if total_stageII == 25 else None
    if desired_train_stageII is None:
        desired_train_stageII = int(round(total_stageII * 13 / 25))
    desired_train = desired_train_stageII
    desired_eval = total_stageII - desired_train

    # To keep class balance, compute class-wise train counts
    class_lists = [zeros, ones]
    class_train_counts = []
    remainders = []
    for cl in class_lists:
        exact = (len(cl) * desired_train) / total_stageII
        floor_count = int(np.floor(exact))
        class_train_counts.append(floor_count)
        remainders.append((exact - floor_count, cl))

    # Distribute remaining train slots according to largest fractional remainder
    current_sum = sum(class_train_counts)
    remaining_to_assign = desired_train - current_sum
    remainders_sorted = sorted(enumerate([r[0] for r in remainders]), key=lambda x: x[1], reverse=True)
    idx_order = [r[0] for r in remainders_sorted]
    k = 0
    while remaining_to_assign > 0:
        class_train_counts[idx_order[k % len(idx_order)]] += 1
        remaining_to_assign -= 1
        k += 1

    # Pick first N from each shuffled class list as train, remainder as eval
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
    test_pts = list(stageI_pts)  # copy

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

    # Flatten indices for DataLoaders (these are image indices)
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

    print(
        f'\nTraining set summary: {len(train_pts)} patients, {len(train_idx)} images. Image counts per class: {train_image_counts}')
    print(
        f'Evaluation set summary: {len(eval_pts)} patients, {len(eval_idx)} images. Image counts per class: {eval_image_counts}')
    print(
        f'Test  set summary: {len(test_pts)} patients, {len(test_idx)} images. Image counts per class: {test_image_counts}')
    print(
        f'Combined Eval+Test summary: {len(comb_pts)} patients, {len(comb_idx)} images. Image counts per class: {comb_image_counts}')

    # Create dataloaders
    batch_size = 64
    train_set = torch.utils.data.Subset(train_data, train_idx)
    eval_set = torch.utils.data.Subset(eval_test_data, eval_idx)
    test_set = torch.utils.data.Subset(eval_test_data, test_idx)
    comb_set = torch.utils.data.Subset(eval_test_data, comb_idx)

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
    models = [ResNet18NPlaned(train_data.shape, start_width=64, n_classes=1)]
    models[len(models):] = [CNNet(train_data.shape),
                            RegularizedCNNet(train_data.shape),
                            ParallelCNNet(train_data.shape),
                            RegularizedParallelCNNet(train_data.shape)]

    for model in models:
        if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
            model.to(device)

    ###################
    # Hyperparameters #
    ###################
    epochs = [250, 500, 1500, 2000, 2500]
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
        with open(f'outputs/results.txt', 'w') as f:
            f.write('Overall Results\n')

    train_loss = [[] for _ in range(len(models))]
    train_auc = [[] for _ in range(len(models))]
    eval_loss = [[] for _ in range(len(models))]
    eval_auc = [[] for _ in range(len(models))]
    best_score = [0 for _ in range(len(models))]

    # For each epoch
    for ep in range(epochs[-1]):
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
            tl.append(el / len(train_set))
            ta.append(roc_auc_score(tx, ot))

        # Evaluation # Train
        epoch_loss = [0 for _ in range(len(models))]

        # Preds and ground truth to calculate training AUC without having to re-run full set
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
                evl.append(el / len(eval_set))
                ea.append(roc_auc_score(tx, ot))

        for i, (model, el, ea, tl) in enumerate(zip(models, eval_loss, eval_auc, train_loss)):
            print(f'>>> {model.name}: Train - Loss: {tl[-1]}. AUC: {ta[-1]}.')
            print(f' --> Eval - Loss: {el[-1]:.4f}. AUC: {ea[-1]:.4f}.')

            with open(f'outputs/{model.name}/results.txt', 'a') as f:
                f.write(f'\nEpoch {ep + 1}'
                        f'>>> {model.name}: Train - Loss: {tl[-1]:.4f}. AUC: {ta[-1]:.4f}.'
                        f'--> Eval - Loss: {el[-1]:.4f}. AUC: {ea[-1]:.4f}.')

            if ea[-1] > best_score[i]:
                best_score[i] = ea[-1]
                torch.save(model.state_dict(), f'outputs/{model.name}/models/Best {model.name}.pth')
                with open(f'outputs/{model.name}/results.txt', 'a') as f:
                    f.write(f'\nNew best {model.name} saved at epoch {ep + 1} with ROC-AUC of {ea[-1]}')
                with open(f'outputs/results.txt', 'a') as f:
                    f.write(f'\nNew best {model.name} saved at epoch {ep + 1} with ROC-AUC of {ea[-1]}')

            if ep + 1 in epochs:
                torch.save(model.state_dict(), f'outputs/{model.name}/models/Epochs {ep + 1} {model.name}.pth')

    with open(f'outputs/results.txt', 'a') as f:
        f.write(f'\n\nFinal ROC-AUC Results\n')
        for model, bs, ta, ea in zip(models, best_score, train_auc, eval_auc):
            f.write(
                f'{model.name}: Best eval AUC - {bs:.4f}. '
                f'Final Train AUC - {ta[-1]:.4f}. Final Eval AUC - {ea[-1]:.4f}\n')

    ####################
    # Testing Section: Updated patient-wise evaluation
    ####################

    def patientwise_score_model(model, dataset, patient_indices, threshold_type='roc'):
        model.eval()
        all_scores = []
        for pt_idx in patient_indices:
            img_idx = dataset.get_patient_subset(pt_idx)
            subset = torch.utils.data.Subset(dataset, img_idx)
            loader = torch.utils.data.DataLoader(subset, batch_size=len(subset), shuffle=False)
            scores, _ = score_model(model, loader, print_results=False, make_plot=False, threshold_type=threshold_type)
            all_scores.append(scores)
        # Average ROC-AUC over patients
        mean_scores = {}
        for key in all_scores[0]:
            if isinstance(all_scores[0][key], (int, float, np.float32, np.float64)):
                mean_scores[key] = float(np.mean([s[key] for s in all_scores]))
        return mean_scores

    headers = ['Best Test', 'Best Eval & Test']
    data = [[] for _ in range(len(models))]
    for i, model in enumerate(models):
        # Best eval model
        model.load_state_dict(torch.load(f'outputs/{model.name}/models/Best {model.name}.pth'))
        print(f'\n>>> {model.name} best evaluated on TEST patients...')
        scores = patientwise_score_model(model, eval_test_data, test_pts, threshold_type='roc')
        data[i].append(scores['ROC-AUC'])
        print(f'ROC-AUC on TEST patients: {scores["ROC-AUC"]:.4f}')

        print(f'\n>>> {model.name} best evaluated on EVAL+TEST patients...')
        scores = patientwise_score_model(model, eval_test_data, comb_pts, threshold_type='roc')
        data[i].append(scores['ROC-AUC'])
        print(f'ROC-AUC on EVAL+TEST patients: {scores["ROC-AUC"]:.4f}')

        # Checkpoint epochs
        for ep in epochs:
            # Load checkpoint
            model.load_state_dict(torch.load(f'outputs/{model.name}/models/Epochs {ep} {model.name}.pth'))
            print(f'\n>>> {model.name} at {ep} epochs on TEST patients...')
            scores = patientwise_score_model(model, eval_test_data, test_pts, threshold_type='roc')
            data[i].append(scores['ROC-AUC'])
            print(f'ROC-AUC: {scores["ROC-AUC"]:.4f}')

            print(f'\n>>> {model.name} at {ep} epochs on EVAL+TEST patients...')
            scores = patientwise_score_model(model, eval_test_data, comb_pts, threshold_type='roc')
            data[i].append(scores['ROC-AUC'])
            print(f'ROC-AUC: {scores["ROC-AUC"]:.4f}')

    auc_table = pd.DataFrame(data=data, index=[model.name for model in models], columns=headers)
    auc_table.to_csv(f'outputs/auc_summary.csv', index_label='Model')

    # Plot and save epoch-wise outputs
    headers = ['Training Loss (average per sample)', 'Evaluation Loss (average per sample)',
               'Training ROC-AUC', 'Evaluation ROC-AUC']
    for (model, tl, el, ta, ea) in zip(models, train_loss, eval_loss, train_auc, eval_auc):
        # Save raw data
        outputs = [[a, b, c, d] for (a, b, c, d) in zip(tl, el, ta, ea)]
        output_table = pd.DataFrame(data=outputs, index=range(1, epochs[-1] + 1), columns=headers)
        output_table.to_csv(f'outputs/{model.name}/tabular.csv', index_label='Epoch')

        # Plot data
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        plt.suptitle(model.name)

        ax1.plot(range(1, epochs[-1] + 1), tl, label=f'{model.name} Training')
        ax1.plot(range(1, epochs[-1] + 1), el, label=f'{model.name} Evaluation')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Evaluation Losses')
        ax1.legend()

        ax2.plot(range(1, epochs[-1] + 1), ta, label=f'{model.name} Training')
        ax2.plot(range(1, epochs[-1] + 1), ea, label=f'{model.name} Evaluation')
        ax1.set_ylabel('Epochs')
        ax2.set_ylabel('AUC')
        ax2.set_title('Training and Evaluation ROC-AUC')
        ax2.legend()

        fig.savefig(f'outputs/{model.name}/plots/losses_and_aucs.png')
        plt.close(fig)

    # Run
    if __name__ == '__main__':
        main()
