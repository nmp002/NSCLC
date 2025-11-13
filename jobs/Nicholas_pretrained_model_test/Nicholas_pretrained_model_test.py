# stageII_pretrained_fixed_split_eval.py
# Load pretrained model (.pth) and evaluate on:
# - Stage II training patients (TRAIN_PTS)
# - Stage II test patients (TEST_PTS_STAGEII)
# - Stage I patients (all Stage I)
# with:
#   * ROC-optimized thresholds per set
#   * Fixed thresholds (0.5788 and 0.5)
#   * Per-image and per-patient CSV reports for each evaluation section.

import os
import random
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from my_modules.models.classifier_models import *
from my_modules.scripts.model_metrics import score_model
from my_modules.scripts.helper_functions import set_seed
from my_modules.scripts.dataset import NSCLCDataset


# ----------------------- Configuration --------------------------
FAST_TEST = False
MODEL_NAME = "ResNet18"

# Update this path to the exact location of your pretrained model file
MODEL_PATH = "/home/nmp002/NSCLC/jobs/Nicholas_pretrained_model_test/models/5-Planed ResNet18_epoch2500.pth"

POOL_METHOD = 'min'  # or 'median'

# Fixed Stage II patient indices (must match dataset patient indices)
TRAIN_PTS = [26, 22, 28, 24, 33, 17, 31, 25, 27, 21, 13, 16, 35, 19, 20, 15, 32]
TEST_PTS_STAGEII = [23, 18, 34, 37, 36, 14, 29, 30]

# Fixed threshold determined earlier from training
FIXED_THRESHOLD_MAIN = 0.5788
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
    method = method.lower()
    if method == 'min':
        return float(np.min(arr))
    if method == 'median':
        return float(np.median(arr))
    # default
    return float(np.median(arr))


def compute_patient_and_image_outputs(model, dataset, patient_indices, device, pool_method='median'):
    """
    For a list of patient indices:
      - run the model on every image
      - compute pooled patient-level scores (min/median)
      - return tensors for score_model + two DataFrames:
        * df_img: one row per image
        * df_pooled: one row per patient (pooled output)
    """
    model.eval()
    image_rows = []
    pooled_rows = []

    with torch.no_grad():
        for pt_idx in patient_indices:
            img_indices = dataset.get_patient_subset(pt_idx)
            if len(img_indices) == 0:
                continue

            name = dataset.get_patient_name(pt_idx)
            label = dataset.get_patient_label(pt_idx).item()
            outs = []

            for im_idx in img_indices:
                x, _ = dataset[im_idx]
                x = x.unsqueeze(0).to(device)
                out = model(x).cpu().detach().squeeze().item()
                outs.append(out)

                image_rows.append({
                    "patient_index": pt_idx,
                    "patient_name": name,
                    "image_index": im_idx,
                    "image_output": out,
                    "label": int(label),
                })

            pooled = pool_patient_scores(outs, method=pool_method)
            pooled_rows.append({
                "patient_index": pt_idx,
                "patient_name": name,
                "pooled_output": pooled,
                "label": int(label),
            })

    if len(pooled_rows) == 0:
        patient_scores = torch.tensor([])
        patient_labels = torch.tensor([])
    else:
        patient_scores = torch.tensor([row["pooled_output"] for row in pooled_rows], dtype=torch.float32)
        patient_labels = torch.tensor([row["label"] for row in pooled_rows], dtype=torch.int64)

    df_img = pd.DataFrame(image_rows)
    df_pooled = pd.DataFrame(pooled_rows)

    return patient_scores, patient_labels, df_img, df_pooled


def main():
    set_seed(42)
    random.seed(42)
    np.random.seed(42)

    print(f"Using model: {MODEL_NAME}")
    print(f"Loading weights from: {MODEL_PATH}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------- Load dataset ------------------------
    data = NSCLCDataset('NSCLC_Data_for_ML',
                        ['fad', 'nadh', 'shg', 'orr', 'intensity'],
                        device=torch.device('cpu'),
                        label='Metastases',
                        mask_on=True)
    data.normalize_method = 'preset'
    data.to(device)

    print("\n==================== LABEL CHECK ====================")

    def check_labels(name, indices):
        print(f"\n{name} (n={len(indices)}):")
        zeros, ones = 0, 0
        for idx in indices:
            label = int(data.get_patient_label(idx).item())
            if label == 0:
                zeros += 1
            else:
                ones += 1
            print(f"  idx {idx:<3}  label={label}   name={data.get_patient_name(idx)}")
        print(f"  --> Class counts: 0={zeros}, 1={ones}")

    # Stage II TRAIN
    check_labels("TRAIN_PTS (Stage II training)", TRAIN_PTS)

    # Stage II TEST
    check_labels("TEST_PTS_STAGEII (Stage II test)", TEST_PTS_STAGEII)

    # Stage I patients
    stageI_indices = [
        i for i in range(data.patient_count)
        if isinstance(data.get_patient_name(i), str)
           and data.get_patient_name(i).endswith('_StageI')
    ]
    check_labels("Stage I patients", stageI_indices)

    # Combined
    combined_indices = TRAIN_PTS + TEST_PTS_STAGEII + stageI_indices
    check_labels("Combined TRAIN + TEST-II + Stage I", combined_indices)

    print("======================================================\n")

    # Stage I patients = any patient whose Slide Name ends with '_StageI'
    stageI_indices = [
        i for i in range(data.patient_count)
        if isinstance(data.get_patient_name(i), str)
        and data.get_patient_name(i).endswith('_StageI')
    ]
    print(f"Found {len(stageI_indices)} Stage I patients for evaluation.")

    # Stage II fixed splits (by index)
    train_stageII_indices = TRAIN_PTS
    stageII_test_indices = TEST_PTS_STAGEII

    print(f"Training Stage II indices: {train_stageII_indices}")
    print(f"Testing Stage II indices: {stageII_test_indices}")

    # ----------------------- Load model --------------------------
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

    out_dir = f"outputs/{MODEL_NAME}/pretrained_fixed_split_pool_{POOL_METHOD}"
    os.makedirs(out_dir, exist_ok=True)

    # ============================================================
    # 1) Stage II TRAINING (indices = TRAIN_PTS), independent ROC
    # ============================================================
    print("\n--- Stage II training evaluation (TRAIN_PTS) ---")

    scores_train_pt, labels_train_pt, df_train_img, df_train_pooled = compute_patient_and_image_outputs(
        model, data, train_stageII_indices, device, pool_method=POOL_METHOD
    )

    # Save per-image and per-patient pooled outputs
    df_train_img.to_csv(os.path.join(out_dir, "train_stageII_image_outputs.csv"), index=False)
    df_train_pooled.to_csv(os.path.join(out_dir, "train_stageII_patient_pooled_outputs.csv"), index=False)

    # Also save the simpler per-patient text file (probabilities)
    with open(os.path.join(out_dir, "train_stageII_probabilities.txt"), 'w') as pf:
        pf.write("Patient Index\tPatient Name\tPooled Probability\tLabel\n")
        for _, row in df_train_pooled.iterrows():
            pf.write(f"{int(row['patient_index'])}\t{row['patient_name']}\t{row['pooled_output']:.4f}\t{int(row['label'])}\n")

    # ROC/PR on training patients (independent ROC-derived threshold)
    scores_train, fig_train = score_model(
        model, (scores_train_pt, labels_train_pt),
        print_results=True, make_plot=True, threshold_type='roc'
    )

    thr_train = scores_train.get('Optimal Threshold from ROC', 0.5)
    print(f"Stage II training ROC-optimized threshold: {thr_train:.4f}")

    fig_train.savefig(os.path.join(out_dir, "train_stageII_combined_ROCopt.png"))
    plt.close(fig_train)

    with open(os.path.join(out_dir, "train_stageII_results.txt"), 'w') as f:
        f.write(f"Stage II training (pool={POOL_METHOD}) — ROC-optimized threshold derived from training set\n")
        f.write(f"Threshold: {thr_train:.4f}\n")
        for key, item in scores_train.items():
            if 'Confusion' not in key:
                f.write(f"|\t{key:<35} {format_metric(item):>10}\t|\n")
        f.write("_____________________________________________________\n")

    # ============================================================
    # 2) Stage II TEST (indices = TEST_PTS_STAGEII), ROC threshold
    # ============================================================
    print("\n--- Stage II test evaluation (TEST_PTS_STAGEII) ---")

    scores_testII_pt, labels_testII_pt, df_testII_img, df_testII_pooled = compute_patient_and_image_outputs(
        model, data, stageII_test_indices, device, pool_method=POOL_METHOD
    )

    # Save per-image and per-patient pooled outputs
    df_testII_img.to_csv(os.path.join(out_dir, "test_stageII_image_outputs.csv"), index=False)
    df_testII_pooled.to_csv(os.path.join(out_dir, "test_stageII_patient_pooled_outputs.csv"), index=False)

    # Also save per-patient probabilities text
    with open(os.path.join(out_dir, "test_stageII_probabilities.txt"), 'w') as pf:
        pf.write("Patient Index\tPatient Name\tPooled Probability\tLabel\n")
        for _, row in df_testII_pooled.iterrows():
            pf.write(f"{int(row['patient_index'])}\t{row['patient_name']}\t{row['pooled_output']:.4f}\t{int(row['label'])}\n")

    scores_testII, fig_testII = score_model(
        model, (scores_testII_pt, labels_testII_pt),
        print_results=True, make_plot=True, threshold_type='roc'
    )

    thr_stageII = scores_testII.get('Optimal Threshold from ROC', 0.5)
    print(f"Stage II ROC-optimized threshold (test set): {thr_stageII:.4f}")

    fig_testII.savefig(os.path.join(out_dir, "test_stageII_combined_ROCopt.png"))
    plt.close(fig_testII)

    with open(os.path.join(out_dir, "test_stageII_results.txt"), 'w') as f:
        f.write(f"Stage II test (pool={POOL_METHOD}) — ROC-optimized threshold derived from Stage II test set\n")
        f.write(f"Threshold: {thr_stageII:.4f}\n")
        for key, item in scores_testII.items():
            if 'Confusion' not in key:
                f.write(f"|\t{key:<35} {format_metric(item):>10}\t|\n")
        f.write("_____________________________________________________\n")

    # ============================================================
    # 3) Stage I TEST (all Stage I), independent ROC threshold
    # ============================================================
    print("\n--- Stage I test evaluation (Independent ROC/Threshold) ---")

    scores_testI_pt, labels_testI_pt, df_testI_img, df_testI_pooled = compute_patient_and_image_outputs(
        model, data, stageI_indices, device, pool_method=POOL_METHOD
    )

    # Save per-image and per-patient pooled outputs
    df_testI_img.to_csv(os.path.join(out_dir, "test_stageI_image_outputs.csv"), index=False)
    df_testI_pooled.to_csv(os.path.join(out_dir, "test_stageI_patient_pooled_outputs.csv"), index=False)

    # Per-patient probabilities text
    with open(os.path.join(out_dir, "test_stageI_probabilities.txt"), 'w') as pf:
        pf.write("Patient Index\tPatient Name\tPooled Probability\tLabel\n")
        for _, row in df_testI_pooled.iterrows():
            pf.write(f"{int(row['patient_index'])}\t{row['patient_name']}\t{row['pooled_output']:.4f}\t{int(row['label'])}\n")

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

    # ============================================================
    # 4) Stage II TEST with fixed threshold = 0.1697
    # ============================================================
    print("\n--- Stage II test evaluation (Fixed threshold = 0.1697) ---")

    scores_testII_fixed, fig_testII_fixed = score_model(
        model, (scores_testII_pt, labels_testII_pt),
        print_results=True, make_plot=True,
        threshold_type='fixed', threshold=FIXED_THRESHOLD_MAIN
    )

    fig_testII_fixed.savefig(os.path.join(out_dir, "test_stageII_combined_fixedThr_0.1697.png"))
    plt.close(fig_testII_fixed)

    with open(os.path.join(out_dir, "test_stageII_results_fixedThr_0.1697.txt"), 'w') as f:
        f.write(f"Stage II test (pool={POOL_METHOD}) — Fixed threshold = {FIXED_THRESHOLD_MAIN:.4f}\n")
        for key, item in scores_testII_fixed.items():
            if 'Confusion' not in key:
                f.write(f"|\t{key:<35} {format_metric(item):>10}\t|\n")
        f.write("_____________________________________________________\n")

    # ============================================================
    # 5) Stage I TEST with fixed threshold = 0.1697
    # ============================================================
    print("\n--- Stage I test evaluation (Fixed threshold = 0.1697) ---")

    scores_testI_fixed, fig_testI_fixed = score_model(
        model, (scores_testI_pt, labels_testI_pt),
        print_results=True, make_plot=True,
        threshold_type='fixed', threshold=FIXED_THRESHOLD_MAIN
    )

    fig_testI_fixed.savefig(os.path.join(out_dir, "test_stageI_combined_fixedThr_0.1697.png"))
    plt.close(fig_testI_fixed)

    with open(os.path.join(out_dir, "test_stageI_results_fixedThr_0.1697.txt"), 'w') as f:
        f.write(f"Stage I test (pool={POOL_METHOD}) — Fixed threshold = {FIXED_THRESHOLD_MAIN:.4f}\n")
        for key, item in scores_testI_fixed.items():
            if 'Confusion' not in key:
                f.write(f"|\t{key:<35} {format_metric(item):>10}\t|\n")
        f.write("_____________________________________________________\n")

    # ============================================================
    # 6) Combined Stage I + Stage II TEST (Fixed Threshold 0.1697)
    # ============================================================
    print("\n--- Combined Stage I + Stage II test evaluation (Fixed Threshold 0.1697) ---")

    combined_indices = stageII_test_indices + stageI_indices
    scores_combined_pt, labels_combined_pt, df_combined_img, df_combined_pooled = compute_patient_and_image_outputs(
        model, data, combined_indices, device, pool_method=POOL_METHOD
    )

    # Save per-image and per-patient pooled outputs
    df_combined_img.to_csv(os.path.join(out_dir, "test_combined_image_outputs_0.1697.csv"), index=False)
    df_combined_pooled.to_csv(os.path.join(out_dir, "test_combined_patient_pooled_outputs_0.1697.csv"), index=False)

    # Simple text probabilities
    with open(os.path.join(out_dir, "test_combined_probabilities_0.1697.txt"), 'w') as pf:
        pf.write("Patient Index\tPatient Name\tPooled Probability\tLabel\n")
        for _, row in df_combined_pooled.iterrows():
            pf.write(f"{int(row['patient_index'])}\t{row['patient_name']}\t{row['pooled_output']:.4f}\t{int(row['label'])}\n")

    scores_combined, fig_combined = score_model(
        model, (scores_combined_pt, labels_combined_pt),
        print_results=True, make_plot=True,
        threshold_type='fixed',
        threshold=FIXED_THRESHOLD_MAIN
    )

    fig_combined.savefig(os.path.join(out_dir, "test_combined_fixedThreshold_0.1697.png"))
    plt.close(fig_combined)

    with open(os.path.join(out_dir, "test_combined_results_fixedThreshold_0.1697.txt"), 'w') as f:
        f.write(f"Combined Stage I + Stage II test (pool={POOL_METHOD}) — Fixed threshold = {FIXED_THRESHOLD_MAIN:.4f}\n")
        for key, item in scores_combined.items():
            if 'Confusion' not in key:
                f.write(f"|\t{key:<35} {format_metric(item):>10}\t|\n")
        f.write("_____________________________________________________\n")

    # ============================================================
    # 7) Combined Stage I + Stage II TEST (Fixed Threshold 0.5)
    # ============================================================
    print("\n--- Combined Stage I + Stage II test evaluation (Fixed Threshold 0.5) ---")

    scores_combined_05, fig_combined_05 = score_model(
        model, (scores_combined_pt, labels_combined_pt),
        print_results=True, make_plot=True,
        threshold_type='fixed',
        threshold=0.5
    )

    fig_combined_05.savefig(os.path.join(out_dir, "test_combined_fixedThreshold_0.5.png"))
    plt.close(fig_combined_05)

    with open(os.path.join(out_dir, "test_combined_results_fixedThreshold_0.5.txt"), 'w') as f:
        f.write(f"Combined Stage I + Stage II test (pool={POOL_METHOD}) — Fixed threshold = 0.5\n")
        for key, item in scores_combined_05.items():
            if 'Confusion' not in key:
                f.write(f"|\t{key:<35} {format_metric(item):>10}\t|\n")
        f.write("_____________________________________________________\n")

    print(f"\nAll results and CSV reports saved to: {out_dir}")


if __name__ == '__main__':
    main()
