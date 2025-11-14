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
POOL_METHOD = 'min'        # 'min' or 'median'
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
    if method == "median":
        return float(np.median(arr))
    return float(np.min(arr))  # default = min pooling


def compute_patient_and_image_outputs(model, dataset, patient_indices, device, pool_method="min"):
    """
    Compute:
      • raw image outputs for each image of each patient
      • pooled patient-level probs
      • return: tensor of pooled probs, tensor of labels, df_img, df_pt
    """
    model.eval()

    img_rows = []
    pt_rows = []

    with torch.no_grad():
        for pt in patient_indices:
            img_idxs = dataset.get_patient_subset(pt)
            if len(img_idxs) == 0:
                continue

            name = dataset.get_patient_name(pt)
            label = int(dataset.get_patient_label(pt).item())
            im_probs = []

            for im in img_idxs:
                x, _ = dataset[im]
                x = x.unsqueeze(0).to(device)
                prob = float(model(x).cpu().detach().item())
                im_probs.append(prob)

                img_rows.append({
                    "patient_index": pt,
                    "patient_name": name,
                    "image_index": im,
                    "image_output": prob,
                    "label": label,
                })

            pooled = pool_patient_scores(im_probs, method=pool_method)

            pt_rows.append({
                "patient_index": pt,
                "patient_name": name,
                "pooled_output": pooled,
                "label": label,
            })

    pt_probs = torch.tensor([r["pooled_output"] for r in pt_rows], dtype=torch.float32)
    pt_labels = torch.tensor([r["label"] for r in pt_rows], dtype=torch.int64)

    return pt_probs, pt_labels, pd.DataFrame(img_rows), pd.DataFrame(pt_rows)


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

        print(f"\nSaved outputs to: {out_dir}")

    print("\n=========================================")
    print("              TESTING COMPLETE")
    print("=========================================\n")


if __name__ == "__main__":
    main()
