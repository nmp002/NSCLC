# stageII_pretrained_test_with_stageI_byID.py
# Load pretrained model (.pth) and test on specific Stage II + Stage I patients.
# Each test set now derives its own ROC-optimized threshold and saves patient probabilities.

import os
import random
import numpy as np
import torch
from matplotlib import pyplot as plt

from my_modules.models.classifier_models import *
from my_modules.scripts.model_metrics import score_model
from my_modules.scripts.helper_functions import set_seed
from my_modules.scripts.dataset import NSCLCDataset


# ----------------------- Configuration --------------------------
FAST_TEST = False
MODEL_NAME = "ResNet18"
MODEL_PATH = "/home/nmp002/NSCLC/jobs/Jesse_pretrained_model_test/models/Epochs 250 4-Planed ResNet18.pth"
POOL_METHOD = 'min'
TEST_PATIENT_IDS_STAGEII = ["S0014", "V0027", "V0142", "S0241", "S0031", "S0093", "V0198", "W0137"]
# ----------------------------------------------------------------


def format_metric(item):
    import torch as _torch
    try:
        if _torch.is_tensor(item) and item.numel() == 1:
            return f"{float(item.item()):.4f}"
        if isinstance(item, (int, float, np.floating, np.integer)):
            return f"{float(item):.4f}"
        return str(item)
    except Exception:
        return str(item)


def pool_patient_scores(outs, method='median'):
    arr = np.asarray(outs, dtype=float)
    if arr.size == 0:
        return float('nan')
    if method == 'min':
        return float(np.min(arr))
    if method == 'median':
        return float(np.median(arr))
    return float(np.median(arr))


def get_patient_indices_by_name(dataset, name_list):
    matched = []
    for i in range(dataset.patient_count):
        name = dataset.get_patient_name(i)
        if not isinstance(name, str):
            continue
        if any(n in name for n in name_list):
            matched.append(i)
    return matched


def patient_wise_loader_outputs(model, dataset, patient_indices, device, pool_method='median'):
    model.eval()
    patient_scores, patient_labels = [], []
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


def main():
    set_seed(42)
    random.seed(42)
    np.random.seed(42)

    print(f"Using model: {MODEL_NAME}")
    print(f"Loading weights from: {MODEL_PATH}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = NSCLCDataset('NSCLC_Data_for_ML',
                        ['fad', 'nadh', 'shg', 'orr'],
                        device=torch.device('cpu'),
                        label='Metastases',
                        mask_on=True)
    data.normalize_method = 'preset'
    data.to(device)

    # Stage II index mapping
    stageII_indices = get_patient_indices_by_name(data, TEST_PATIENT_IDS_STAGEII)
    patient_names = {i: data.get_patient_name(i) for i in stageII_indices}
    print(f"Resolved Stage II test patients: {[(i, patient_names[i]) for i in stageII_indices]}")
    if len(stageII_indices) == 0:
        raise ValueError("No Stage II test patients found. Check naming convention or ID list.")
    if len(stageII_indices) != len(TEST_PATIENT_IDS_STAGEII):
        print("Warning: Some specified patient IDs were not matched exactly.")

    # Stage I patients
    stageI_indices = [
        i for i in range(data.patient_count)
        if isinstance(data.get_patient_name(i), str) and data.get_patient_name(i).endswith('_StageI')
    ]
    print(f"Found {len(stageI_indices)} Stage I patients for evaluation.")

    # Load model
    if MODEL_NAME == "ResNet18":
        model = ResNet18NPlaned(data.shape, start_width=64, n_classes=1)
    elif MODEL_NAME == "CNNet":
        model = CNNet(data.shape)
    elif MODEL_NAME == "RegularizedCNNet":
        model = RegularizedCNNet(data.shape)
    else:
        raise ValueError(f"Unknown model name: {MODEL_NAME}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    out_dir = f"outputs/{MODEL_NAME}/pretrained_test_pool_{POOL_METHOD}"
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- Stage II TEST ----------------
    print("\n--- Stage II test evaluation ---")
    scores_testII_pt, labels_testII_pt = patient_wise_loader_outputs(model, data, stageII_indices, device, pool_method=POOL_METHOD)

    # Save per-patient probabilities
    with open(os.path.join(out_dir, "test_stageII_probabilities.txt"), 'w') as pf:
        pf.write("Patient Index\tPatient Name\tProbability\tLabel\n")
        for idx, score, label in zip(stageII_indices, scores_testII_pt, labels_testII_pt):
            name = data.get_patient_name(idx)
            pf.write(f"{idx}\t{name}\t{score.item():.4f}\t{int(label)}\n")

    scores_testII, fig_testII = score_model(
        model, (scores_testII_pt, labels_testII_pt),
        print_results=True, make_plot=True, threshold_type='roc'
    )

    thr_stageII = scores_testII.get('Optimal Threshold from ROC', 0.5)
    print(f"Stage II ROC-optimized threshold: {thr_stageII:.4f}")

    fig_testII.savefig(os.path.join(out_dir, "test_stageII_combined_ROCopt.png"))
    plt.close(fig_testII)

    with open(os.path.join(out_dir, "test_stageII_results.txt"), 'w') as f:
        f.write(f"Stage II test (pool={POOL_METHOD}) — ROC-optimized threshold derived from Stage II\n")
        f.write(f"Threshold: {thr_stageII:.4f}\n")
        for key, item in scores_testII.items():
            if 'Confusion' not in key:
                f.write(f"|\t{key:<35} {format_metric(item):>10}\t|\n")
        f.write("_____________________________________________________\n")

    # ---------------- Stage I TEST (Independent ROC) ----------------
    print("\n--- Stage I test evaluation (Independent ROC/Threshold) ---")
    scores_testI_pt, labels_testI_pt = patient_wise_loader_outputs(model, data, stageI_indices, device, pool_method=POOL_METHOD)

    # Save per-patient probabilities
    with open(os.path.join(out_dir, "test_stageI_probabilities.txt"), 'w') as pf:
        pf.write("Patient Index\tPatient Name\tProbability\tLabel\n")
        for idx, score, label in zip(stageI_indices, scores_testI_pt, labels_testI_pt):
            name = data.get_patient_name(idx)
            pf.write(f"{idx}\t{name}\t{score.item():.4f}\t{int(label)}\n")

    scores_testI, fig_testI = score_model(
        model, (scores_testI_pt, labels_testI_pt),
        print_results=True, make_plot=True, threshold_type='roc'
    )

    thr_stageI = scores_testI.get('Optimal Threshold from ROC', 0.5)
    print(f"Stage I ROC-optimized threshold: {thr_stageI:.4f}")

    fig_testI.savefig(os.path.join(out_dir, "test_stageI_combined_independentROC.png"))
    plt.close(fig_testI)

    with open(os.path.join(out_dir, "test_stageI_results_independentROC.txt"), 'w') as f:
        f.write(f"Stage I test (pool={POOL_METHOD}) — ROC-optimized threshold derived from Stage I\n")
        f.write(f"Threshold: {thr_stageI:.4f}\n")
        for key, item in scores_testI.items():
            if 'Confusion' not in key:
                f.write(f"|\t{key:<35} {format_metric(item):>10}\t|\n")
        f.write("_____________________________________________________\n")

    print(f"\nResults saved to: {out_dir}")


if __name__ == '__main__':
    main()
