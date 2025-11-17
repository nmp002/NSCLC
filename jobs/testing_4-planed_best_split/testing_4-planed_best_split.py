# stageII_pretrained_trainset_patientwise_test.py
# ---------------------------------------------------------
# Load ALL pretrained models (*.pth) in ./models/
# Evaluate ONLY on Stage II TRAINING SET (TRAIN_PTS)
#   • Patient-wise pooled probabilities (min or median)
#   • ROC-optimized threshold from training
#   • Per-image + per-patient CSV reports
#   • One output folder per model
# ---------------------------------------------------------

import os
import random
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from my_modules.models.classifier_models import (
    ResNet18NPlaned, CNNet, RegularizedCNNet,
    ParallelCNNet, RegularizedParallelCNNet
)
from my_modules.scripts.model_metrics import score_model
from my_modules.scripts.helper_functions import set_seed
from my_modules.scripts.dataset import NSCLCDataset


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
POOL_METHOD = 'max'        # 'min', 'max' or 'median'
MODELS_DIR = "/home/nmp002/NSCLC/jobs/testing_4-planed_best_split/models/"      # where your *.pth files are stored

# Training split EXACTLY as before
TRAIN_PTS = [26, 22, 28, 24, 33, 17, 31, 25, 27, 21, 13, 16, 35, 19, 20, 15, 32]


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

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


def pool_patient_scores(prob_list, method="min"):
    arr = np.asarray(prob_list, dtype=float)
    if arr.size == 0:
        return float("nan")

    method = method.lower()

    if method == "median":
        return float(np.median(arr))
    if method == "max":
        return float(np.max(arr))
    return float(np.min(arr))    # default = min



def compute_patient_and_image_outputs(model, dataset, patient_indices, device, pool_method="min"):
    """
    EXACT BEHAVIOR from stageII_splits_min_median.py:
        • One row per patient (not per image)
        • 'image_outputs' column lists ALL FOV outputs as semicolon-separated strings
        • Pooled patient score (min or median)
    """

    model.eval()

    patient_probs = []
    patient_labels = []
    rows = []

    with torch.no_grad():
        for pt_idx in patient_indices:

            img_indices = dataset.get_patient_subset(pt_idx)
            if len(img_indices) == 0:
                continue

            name = dataset.get_patient_name(pt_idx)
            label = int(dataset.get_patient_label(pt_idx).item())

            outs = []
            for im_idx in img_indices:
                x, _ = dataset[im_idx]
                x = x.unsqueeze(0).to(device)
                prob = float(model(x).cpu().detach().item())
                outs.append(prob)

            pooled = pool_patient_scores(outs, method=pool_method)

            patient_probs.append(pooled)
            patient_labels.append(label)

            rows.append({
                "patient_index": int(pt_idx),
                "patient_name": str(name),
                "label": label,
                "n_images": len(outs),
                "image_outputs": ";".join([f"{v:.6f}" for v in outs]),
                "pool_method": pool_method,
                "pooled_output": f"{pooled:.6f}",
            })

    df_img = pd.DataFrame(rows)     # one row per patient
    df_pt = df_img.copy()           # identical for compatibility

    pt_probs = torch.tensor(patient_probs, dtype=torch.float32)
    pt_labels = torch.tensor(patient_labels, dtype=torch.int64)

    return pt_probs, pt_labels, df_img, df_pt



def instantiate_model_by_name(name, data_shape):
    """
    Infer model type by filename substring.
    Modify as needed if you have different naming conventions.
    """
    name_lower = name.lower()

    if "resnet18" in name_lower:
        return ResNet18NPlaned(data_shape, start_width=64, n_classes=1)

    if "regularizedparallel" in name_lower:
        return RegularizedParallelCNNet(data_shape)

    if "parallel" in name_lower:
        return ParallelCNNet(data_shape)

    if "regularized" in name_lower:
        return RegularizedCNNet(data_shape)

    if "cnnet" in name_lower:
        return CNNet(data_shape)

    # default fallback
    return ResNet18NPlaned(data_shape, start_width=64, n_classes=1)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    set_seed(42)
    random.seed(42)
    np.random.seed(42)

    print("\n=========================================")
    print("       PATIENT-WISE TRAINING TEST")
    print("=========================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset (same as training)
    print("Loading dataset ...")
    data = NSCLCDataset(
        'NSCLC_Data_for_ML',
        ['fad', 'nadh', 'shg', 'orr'],
        device=torch.device('cpu'),
        label='Metastases',
        mask_on=True
    )

    data.augment()
    data.normalize_method = 'preset'
    data.to(device)

    print(f"Training patients (n={len(TRAIN_PTS)}): {TRAIN_PTS}")

    # Find all *.pth models
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]
    if len(model_files) == 0:
        print("\nNo models found in ./models/")
        return

    print(f"\nFound {len(model_files)} model files.\n")

    # Loop through all saved models
    for model_file in model_files:
        model_path = os.path.join(MODELS_DIR, model_file)
        model_name = os.path.splitext(model_file)[0]

        print("\n-----------------------------------------")
        print(f"Testing model: {model_name}")
        print("-----------------------------------------")

        # Create output directory
        out_dir = f"outputs/{model_name}/trainset_patientwise_pool_{POOL_METHOD}"
        os.makedirs(out_dir, exist_ok=True)

        # Instantiate model based on filename
        model = instantiate_model_by_name(model_name, data.shape)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # -----------------------------------------
        # PATIENT-WISE INFERENCE ON TRAINING SET
        # -----------------------------------------
        pt_probs, pt_labels, df_img, df_pooled = compute_patient_and_image_outputs(
            model, data, TRAIN_PTS, device, pool_method=POOL_METHOD
        )

        # Save CSVs
        df_img.to_csv(os.path.join(out_dir, "train_image_outputs.csv"), index=False)
        df_pooled.to_csv(os.path.join(out_dir, "train_patient_pooled_outputs.csv"), index=False)

        # -----------------------------------------
        # ROC THRESHOLD FROM TRAINING SET
        # -----------------------------------------
        scores, fig = score_model(
            model, (pt_probs, pt_labels),
            print_results=True, make_plot=True,
            threshold_type='roc'
        )

        thr_train = scores.get("Optimal Threshold from ROC", 0.5)

        # Save ROC plot and results
        fig.savefig(os.path.join(out_dir, "train_ROC.png"))
        plt.close(fig)

        with open(os.path.join(out_dir, "train_results.txt"), "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Pooling: {POOL_METHOD}\n")
            f.write(f"Training ROC-optimized threshold: {thr_train:.4f}\n\n")
            for key, item in scores.items():
                if "Confusion" not in key:
                    f.write(f"{key:<40} {format_metric(item)}\n")

        # =========================================================
        # CONSTANT THRESHOLD TESTING: 0.5
        # =========================================================

        CONST_THR = 0.5

        # -----------------------------
        # Stage II Test Set (held-out)
        # -----------------------------
        TEST_PTS_STAGEII = [23, 18, 34, 37, 36, 14, 29, 30]
        out_stageII = os.path.join(out_dir, "const_thresh_0.5_stageII")
        os.makedirs(out_stageII, exist_ok=True)

        pt_probs, pt_labels, df_img, df_pt = compute_patient_and_image_outputs(
            model, data, TEST_PTS_STAGEII, device, pool_method=POOL_METHOD
        )

        # save CSVs
        df_img.to_csv(os.path.join(out_stageII, "image_outputs.csv"), index=False)
        df_pt.to_csv(os.path.join(out_stageII, "patient_outputs.csv"), index=False)

        # compute accuracy, sens, spec at constant threshold
        preds = (pt_probs.numpy() >= CONST_THR).astype(int)
        labels = pt_labels.numpy()

        acc  = np.mean(preds == labels)
        sens = np.sum((preds == 1) & (labels == 1)) / max(np.sum(labels == 1), 1)
        spec = np.sum((preds == 0) & (labels == 0)) / max(np.sum(labels == 0), 1)

        with open(os.path.join(out_stageII, "results.txt"), "w") as f:
            f.write(f"Constant threshold: {CONST_THR}\n")
            f.write(f"N patients: {len(labels)}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Sensitivity: {sens:.4f}\n")
            f.write(f"Specificity: {spec:.4f}\n")


        # -----------------------------
        # Stage I Test Set
        # Stage I = everything NOT listed in TRAIN_PTS or TEST_PTS_STAGEII
        # -----------------------------
        all_pts = list(range(data.patient_count))  # all rows in Excel file
        STAGE_I_PTS = [
            p for p in all_pts if (p not in TRAIN_PTS) and (p not in TEST_PTS_STAGEII)
        ]

        out_stageI = os.path.join(out_dir, "const_thresh_0.5_stageI")
        os.makedirs(out_stageI, exist_ok=True)

        pt_probs, pt_labels, df_img, df_pt = compute_patient_and_image_outputs(
            model, data, STAGE_I_PTS, device, pool_method=POOL_METHOD
        )

        df_img.to_csv(os.path.join(out_stageI, "image_outputs.csv"), index=False)
        df_pt.to_csv(os.path.join(out_stageI, "patient_outputs.csv"), index=False)

        preds = (pt_probs.numpy() >= CONST_THR).astype(int)
        labels = pt_labels.numpy()

        acc  = np.mean(preds == labels)
        sens = np.sum((preds == 1) & (labels == 1)) / max(np.sum(labels == 1), 1)
        spec = np.sum((preds == 0) & (labels == 0)) / max(np.sum(labels == 0), 1)

        with open(os.path.join(out_stageI, "results.txt"), "w") as f:
            f.write(f"Constant threshold: {CONST_THR}\n")
            f.write(f"N patients: {len(labels)}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Sensitivity: {sens:.4f}\n")
            f.write(f"Specificity: {spec:.4f}\n")


        # -----------------------------
        # Stage I + Stage II Combined
        # -----------------------------
        COMBINED = STAGE_I_PTS + TEST_PTS_STAGEII

        out_comb = os.path.join(out_dir, "const_thresh_0.5_stageI_and_stageII")
        os.makedirs(out_comb, exist_ok=True)

        pt_probs, pt_labels, df_img, df_pt = compute_patient_and_image_outputs(
            model, data, COMBINED, device, pool_method=POOL_METHOD
        )

        df_img.to_csv(os.path.join(out_comb, "image_outputs.csv"), index=False)
        df_pt.to_csv(os.path.join(out_comb, "patient_outputs.csv"), index=False)

        preds = (pt_probs.numpy() >= CONST_THR).astype(int)
        labels = pt_labels.numpy()

        acc  = np.mean(preds == labels)
        sens = np.sum((preds == 1) & (labels == 1)) / max(np.sum(labels == 1), 1)
        spec = np.sum((preds == 0) & (labels == 0)) / max(np.sum(labels == 0), 1)

        with open(os.path.join(out_comb, "results.txt"), "w") as f:
            f.write(f"Constant threshold: {CONST_THR}\n")
            f.write(f"N patients: {len(labels)}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Sensitivity: {sens:.4f}\n")
            f.write(f"Specificity: {spec:.4f}\n")


        print(f"\nSaved outputs to: {out_dir}")

    print("\n=========================================")
    print("              TESTING COMPLETE")
    print("=========================================\n")


if __name__ == "__main__":
    main()
