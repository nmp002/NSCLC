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
    ones = [i for i, l in paired if int(l) == 1]   # non-metastatic

    print(f'Stage II metastatic (label=0): {len(zeros)} patients')
    print(f'Stage II non-metastatic (label=1): {len(ones)} patients')

    # Desired Stage II train/eval sizes (you specified: split 25 StageII -> 13 / 12)
    total_stageII = len(stageII_pts)
    desired_train_stageII = int(round(total_stageII * 13 / 25)) if total_stageII == 25 else None
    # If the user specifically expects 13/12 when total_stageII == 25, enforce it; otherwise default to same ratio as original:
    if desired_train_stageII is None:
        # fallback to original ratio from your script:
        # original used eval first 3 from each class then rest as train (not directly transferable),
        # simpler fallback: use 0.52 train fraction ~ 13/25
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
    # sort remainders descending by fractional part
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

    # Test set is ALL Stage I patients (explicit exclusion from training) - user requested this.
    test_pts = list(stageI_pts)  # copy
    # If you still want to include any Stage II holdouts in the test set (you previously had last-of-class),
    # the user requested *all Stage I* to be test, and Stage II split into train/eval, so we DO NOT add StageII patients to test.

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

    # Image count summaries (optional, mirror previous print style)
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

    # 1. How many total patient indices?
    print("Total patients detected:", train_data.patient_count)

    # 2. Print FIRST 10 raw patient names from get_patient_name():
    print("\nSample patient names from dataset:")
    for i in range(min(10, train_data.patient_count)):
        print(f"  idx {i}: {train_data.get_patient_name(i)}")

    # 3. Check if ANY patients have 'Stage' in their name at all
    stage_like = [train_data.get_patient_name(i) for i in range(train_data.patient_count) if "Stage" in train_data.get_patient_name(i)]
    print("\nPatients with 'Stage' in name:", stage_like)



# Run
if __name__ == '__main__':
    main()
