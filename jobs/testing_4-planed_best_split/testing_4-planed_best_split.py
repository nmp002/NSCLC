# stageII_pretrained_trainset_patientwise_test.py
# ---------------------------------------------------------
# Load ALL pretrained models (*.pth) in ./models/
# Evaluate:
#   1. Training set (ROC-optimized threshold)
#   2. Stage II test @ 0.5 threshold
#   3. Stage I test @ 0.5 threshold
#   4. Combined Stage I + Stage II test @ 0.5 threshold
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
POOL_METHOD = 'median'        # 'min', 'max', 'majority', 'mean', or 'median'
MODELS_DIR = "/home/nmp002/NSCLC/jobs/testing_4-planed_best_split/models/"

TRAIN_PTS = [26, 22, 28, 24, 33, 17, 31, 25,
             27, 21, 13, 16, 35, 19, 20, 15, 32]

TEST_PTS_STAGEII = [23, 18, 34, 37, 36, 14, 29, 30]


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

    # --------- MEAN / AVERAGE POOLING ---------
    if method in ("mean", "avg"):
        return float(np.mean(arr))

    # --------- MEDIAN POOLING ---------
    if method == "median":
        return float(np.median(arr))

    # --------- MAX POOLING ---------
    if method == "max":
        return float(np.max(arr))

    # --------- MAJORITY VOTE POOLING ---------
    if method in ("majority", "vote", "mv"):
        # Convert probs → hard predictions
        preds = (arr >= 0.5).astype(int)
        ones = preds.sum()
        zeros = len(preds) - ones

        # If tie, choose 1 (can be changed)
        return 1.0 if ones >= zeros else 0.0

    # --------- DEFAULT = MIN POOLING ---------
    return float(np.min(arr))



def compute_patient_and_image_outputs(model, dataset, patient_indices, device, pool_method="min"):
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

    df_img = pd.DataFrame(rows)
    df_pt = df_img.copy()

    pt_probs = torch.tensor(patient_probs, dtype=torch.float32)
    pt_labels = torch.tensor(patient_labels, dtype=torch.int64)

    return pt_probs, pt_labels, df_img, df_pt


def instantiate_model_by_name(name, data_shape):
    n = name.lower()
    if "resnet18" in n:
        return ResNet18NPlaned(data_shape, start_width=64, n_classes=1)
    if "regularizedparallel" in n:
        return RegularizedParallelCNNet(data_shape)
    if "parallel" in n:
        return ParallelCNNet(data_shape)
    if "regularized" in n:
        return RegularizedCNNet(data_shape)
    if "cnnet" in n:
        return CNNet(data_shape)
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

    # Load dataset
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

    # ---------------------------------------------------------
    # Load model list
    # ---------------------------------------------------------
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]
    if len(model_files) == 0:
        print("\nNo models found.")
        return

    print(f"\nFound {len(model_files)} model files.\n")

    # ---------------------------------------------------------
    # MAIN LOOP OVER MODELS
    # ---------------------------------------------------------
    for model_file in model_files:
        model_path = os.path.join(MODELS_DIR, model_file)
        model_name = os.path.splitext(model_file)[0]

        print("\n-----------------------------------------")
        print(f"Testing model: {model_name}")
        print("-----------------------------------------")

        out_dir = f"outputs/{model_name}/trainset_patientwise_pool_{POOL_METHOD}"
        os.makedirs(out_dir, exist_ok=True)

        model = instantiate_model_by_name(model_name, data.shape)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # =========================================================
        # TRAINING SET PATIENT-WISE TEST
        # =========================================================
        pt_probs, pt_labels, df_img, df_pooled = compute_patient_and_image_outputs(
            model, data, TRAIN_PTS, device, pool_method=POOL_METHOD
        )

        df_img.to_csv(os.path.join(out_dir, "train_image_outputs.csv"), index=False)
        df_pooled.to_csv(os.path.join(out_dir, "train_patient_pooled_outputs.csv"), index=False)

        scores, fig = score_model(
            model, (pt_probs, pt_labels),
            print_results=True,
            make_plot=True,
            threshold_type='roc'
        )

        thr_train = scores.get("Optimal Threshold from ROC", 0.5)

        fig.savefig(os.path.join(out_dir, "train_ROC.png"))
        plt.close(fig)

        with open(os.path.join(out_dir, "train_results.txt"), "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Pooling: {POOL_METHOD}\n")
            f.write(f"Training ROC-optimized threshold: {thr_train:.4f}\n\n")
            for key, item in scores.items():
                if "Confusion" not in key:
                    f.write(f"{key:<40} {format_metric(item)}\n")

        # ============================================================
        # CONSTANT THRESHOLD (0.5) SECTIONS
        # ============================================================

        CONST_THR = 0.5

        # -------- Stage I IDs --------
        ALL_PTS = list(range(data.patient_count))
        STAGE_I_PTS = [
            p for p in ALL_PTS
            if ("StageI" in str(data.get_patient_name(p)))
        ]

        TEST_SETS = {
            "const_thr0p5_stageII": TEST_PTS_STAGEII,
            "const_thr0p5_stageI": STAGE_I_PTS,
            "const_thr0p5_combined": TEST_PTS_STAGEII + STAGE_I_PTS
        }

        for test_name, pt_list in TEST_SETS.items():

            out_test = os.path.join(out_dir, test_name)
            os.makedirs(out_test, exist_ok=True)

            pt_probs, pt_labels, df_img_test, df_pt_test = compute_patient_and_image_outputs(
                model, data, pt_list, device, pool_method=POOL_METHOD
            )

            df_img_test.to_csv(os.path.join(out_test, "image_outputs.csv"), index=False)
            df_pt_test.to_csv(os.path.join(out_test, "patient_outputs.csv"), index=False)

            # ROC curve using score_model
            scores_test, fig_roc = score_model(
                model,
                (pt_probs, pt_labels),
                print_results=True,
                make_plot=True,
                threshold_type='fixed'
            )
            fig_roc.savefig(os.path.join(out_test, "ROC_curve.png"))
            plt.close(fig_roc)

            # confusion matrix @ 0.5
            preds = (pt_probs >= CONST_THR).long()

            TP = int(((preds == 1) & (pt_labels == 1)).sum())
            TN = int(((preds == 0) & (pt_labels == 0)).sum())
            FP = int(((preds == 1) & (pt_labels == 0)).sum())
            FN = int(((preds == 0) & (pt_labels == 1)).sum())

            cm = np.array([[TN, FP], [FN, TP]])

            fig_cm = plt.figure(figsize=(4, 4))
            ax = fig_cm.add_subplot(111)
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"])
            ax.set_yticklabels(["True 0", "True 1"])
            ax.set_title(f"Confusion Matrix @0.5\n({test_name})")

            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center")

            fig_cm.savefig(os.path.join(out_test, "confusion_matrix_thr0p5.png"))
            plt.close(fig_cm)

            # metrics file
            acc = float((preds == pt_labels).float().mean())
            sens = TP / (TP + FN + 1e-9)
            spec = TN / (TN + FP + 1e-9)

            with open(os.path.join(out_test, "results_thr0p5.txt"), "w") as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Test Set: {test_name}\n")
                f.write(f"Pooling: {POOL_METHOD}\n")
                f.write(f"Threshold: 0.5\n\n")
                f.write(f"Accuracy:       {acc:.4f}\n")
                f.write(f"Sensitivity:    {sens:.4f}\n")
                f.write(f"Specificity:    {spec:.4f}\n\n")

                f.write("ROC-based metrics:\n")
                for key, val in scores_test.items():
                    if "Confusion" not in key:
                        f.write(f"{key:<40} {format_metric(val)}\n")

        print(f"\nSaved outputs to: {out_dir}")

    print("\n=========================================")
    print("              TESTING COMPLETE")
    print("=========================================\n")


if __name__ == "__main__":
    main()
