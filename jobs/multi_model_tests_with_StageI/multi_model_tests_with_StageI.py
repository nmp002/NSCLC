import random
import numpy as np
import torch
from my_modules.scripts.dataset import NSCLCDataset

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Prepare data
    train_data = NSCLCDataset(
        'NSCLC_Data_for_ML',
        ['fad', 'nadh', 'shg', 'intensity', 'orr'],
        device=torch.device('cpu'),
        label='Metastases',
        mask_on=True
    )
    train_data.augment()
    train_data.normalize_method = 'preset'

    # Get all patient indices
    idx = list(range(train_data.patient_count))

    # Filter out patients with no images
    patient_subsets = [train_data.get_patient_subset(i) for i in idx]
    idx = [i for i, subset in zip(idx, patient_subsets) if len(subset) > 0]

    # Stage I vs Stage II detection
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

    print(f'Found {len(stageII_pts)} Stage II patients and {len(stageI_pts)} Stage I patients.\n')
    print('Stage I patient names (TEST set):')
    for i in stageI_pts:
        print(f'  - idx {i}: {patient_names[i]}')
    print('\nStage II patient names (to be split into TRAIN / EVAL):')
    for i in stageII_pts:
        print(f'  - idx {i}: {patient_names[i]}')

    # Shuffle Stage II patients
    random.shuffle(stageII_pts)
    # Simple 50/50 split for testing
    split_idx = len(stageII_pts) // 2
    train_pts = stageII_pts[:split_idx]
    eval_pts = stageII_pts[split_idx:]

    test_pts = list(stageI_pts)  # Stage I patients are test

    print('\nFINAL SPLITS:')
    print(f'  TRAIN (n={len(train_pts)}): {[patient_names[i] for i in train_pts]}')
    print(f'  EVAL  (n={len(eval_pts)}): {[patient_names[i] for i in eval_pts]}')
    print(f'  TEST  (Stage I) (n={len(test_pts)}): {[patient_names[i] for i in test_pts]}')

if __name__ == '__main__':
    main()
