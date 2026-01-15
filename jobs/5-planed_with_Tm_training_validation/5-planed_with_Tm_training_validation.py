# 5-planed_with_Tm_training_validation.py
# Train 5-planed ResNet18 variant on fixed 17 Stage II patients,
# with held-out 8 Stage II patients as validation set.
# Inputs: ['fad', 'nadh', 'shg', 'orr', 'Tm'] (5 planes).
# Train for 2500 epochs, save model and train/val loss+AUC curves every 250 epochs.
# No ROC/threshold/confusion-matrix evaluation.

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms.v2 as tvt
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
import torch.nn as nn

from my_modules.models.classifier_models import ResNet18NPlaned
from my_modules.scripts.helper_functions import set_seed
from my_modules.scripts.dataset2 import NSCLCDataset

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
FAST_TEST = False            # if True: small epochs etc. for smoke testing
TOTAL_EPOCHS = 2500          # train for 2500 epochs
SAVE_INTERVAL = 500          # save models and plots every 250 epochs

# fixed Stage II patient indices (must match dataset)
TRAIN_PTS = [26, 22, 28, 24, 33, 17, 31, 25, 27, 21, 13, 16, 35, 19, 20, 15, 32]
TEST_PTS_STAGEII = [23, 18, 34, 37, 36, 14, 29, 30]

LR = 1e-7
WD = 0.2


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    # seeds
    set_seed(42)
    random.seed(42)
    np.random.seed(42)

    print(f"Num cores: {mp.cpu_count()}")
    print(f"Num GPUs: {torch.cuda.device_count()}")

    try:
        mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Data  (5 modes: fad, nadh, shg, orr, Tm)
    # ------------------------------------------------------------------
    modes = ["fad", "nadh", "shg", "orr", "tm"]

    train_data = NSCLCDataset(
        "NSCLC_Data_for_ML",
        modes,
        device=torch.device("cpu"),
        label="Metastases",
        mask_on=True,
        remove_empties=False
    )
    eval_data = NSCLCDataset(
        "NSCLC_Data_for_ML",
        modes,
        device=torch.device("cpu"),
        label="Metastases",
        mask_on=True,
        remove_empties=False
    )

    if FAST_TEST:
        train_data.augmented = False
        eval_data.augmented = False
        train_data.normalize_method = "preset"
        eval_data.normalize_method = "preset"
        train_data.transforms = None
        eval_data.transforms = None
        train_data.to(device)
        eval_data.to(device)
        total_epochs = 3
        batch_size = 8
    else:
        train_data.augment()
        train_data.normalize_method = "preset"
        train_data.to(device)
        train_data.transforms = tvt.Compose(
            [
                tvt.RandomVerticalFlip(p=0.25),
                tvt.RandomHorizontalFlip(p=0.25),
                tvt.RandomRotation(degrees=(-180, 180)),
            ]
        )

        eval_data.normalize_method = "preset"
        eval_data.to(device)
        total_epochs = TOTAL_EPOCHS
        batch_size = 64

    # flatten image indices for training and Stage II validation
    train_img_idx = [train_data.get_patient_subset(i) for i in TRAIN_PTS]
    train_img_idx = [im for sub in train_img_idx for im in sub]
    random.shuffle(train_img_idx)

    val_img_idx_stageII = [eval_data.get_patient_subset(i) for i in TEST_PTS_STAGEII]
    val_img_idx_stageII = [im for sub in val_img_idx_stageII for im in sub]
    random.shuffle(val_img_idx_stageII)

    # dataloaders (image-wise)
    train_set = torch.utils.data.Subset(train_data, train_img_idx)
    val_set_stageII = torch.utils.data.Subset(eval_data, val_img_idx_stageII)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader_stageII = torch.utils.data.DataLoader(
        val_set_stageII,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # Model (only 5-planed ResNet18 variant)
    # ------------------------------------------------------------------
    model = ResNet18NPlaned(train_data.shape, start_width=64, n_classes=1)
    if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
        model.to(device)

    # ------------------------------------------------------------------
    # Training setup
    # ------------------------------------------------------------------
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    # output dirs and files
    os.makedirs("outputs", exist_ok=True)
    os.makedirs(f"outputs/{model.name}/plots", exist_ok=True)
    os.makedirs(f"outputs/{model.name}/models", exist_ok=True)

    train_loss = []
    train_auc = []
    val_loss = []
    val_auc = []

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for ep in range(total_epochs):
        print(f"\nEpoch {ep + 1}/{total_epochs}")

        # ---- Training ----
        model.train()
        epoch_train_loss = 0.0
        train_outs = torch.tensor([])
        train_targets = torch.tensor([])

        for x, target in train_loader:
            x, target = x.to(device), target.to(device)
            out = model(x)  # (B, 1)
            loss = loss_fn(out, target.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_outs = torch.cat((train_outs, out.detach().cpu()), dim=0)
            train_targets = torch.cat((train_targets, target.detach().cpu()), dim=0)

        # normalize loss by number of samples
        epoch_train_loss /= max(len(train_set), 1)
        train_loss.append(epoch_train_loss)

        try:
            epoch_train_auc = roc_auc_score(train_targets.numpy(), train_outs.numpy())
        except Exception:
            epoch_train_auc = 0.0
        train_auc.append(epoch_train_auc)

        print(f">>> {model.name}: Train Loss={epoch_train_loss:.4f}, Train AUC={epoch_train_auc:.4f}")

        # ---- Validation (Stage II test set used as validation) ----
        model.eval()
        epoch_val_loss = 0.0
        val_outs = torch.tensor([])
        val_targets = torch.tensor([])

        with torch.no_grad():
            for x_val, target_val in val_loader_stageII:
                x_val, target_val = x_val.to(device), target_val.to(device)
                out_val = model(x_val)
                loss_val = loss_fn(out_val, target_val.unsqueeze(1))

                epoch_val_loss += loss_val.item()
                val_outs = torch.cat((val_outs, out_val.detach().cpu()), dim=0)
                val_targets = torch.cat((val_targets, target_val.detach().cpu()), dim=0)

        epoch_val_loss /= max(len(val_set_stageII), 1)
        val_loss.append(epoch_val_loss)

        try:
            epoch_val_auc = roc_auc_score(val_targets.numpy(), val_outs.numpy())
        except Exception:
            epoch_val_auc = 0.0
        val_auc.append(epoch_val_auc)

        print(f">>> {model.name}: Val Loss={epoch_val_loss:.4f}, Val AUC={epoch_val_auc:.4f}")

        # ---- Save model + plots every SAVE_INTERVAL epochs ----
        epoch_num = ep + 1
        if (epoch_num % SAVE_INTERVAL == 0) or (epoch_num == total_epochs):
            # Save model
            model_path = (
                f"outputs/{model.name}/models/{model.name}_lr_{LR}_wd_{WD}_epoch{epoch_num}.pth"
            )
            torch.save(model.state_dict(), model_path)
            print(f"Saved model checkpoint: {model_path}")

            # Save curves up to current epoch
            df = pd.DataFrame(
                {
                    "Training Loss": train_loss,
                    "Validation Loss": val_loss,
                    "Training ROC-AUC": train_auc,
                    "Validation ROC-AUC": val_auc,
                },
                index=range(1, len(train_loss) + 1),
            )

            df.to_csv(
                f"outputs/{model.name}/tabular_train_val_lr_{LR}_wd_{WD}.csv",
                index_label="Epoch",
            )

            # Plot loss and AUC with train vs val overlaid
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
            plt.suptitle(f"{model.name} (up to epoch {epoch_num})")

            # Loss
            ax1.plot(df.index, df["Training Loss"], label="Train Loss")
            ax1.plot(df.index, df["Validation Loss"], label="Val Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training vs Validation Loss")
            ax1.legend()

            # AUC
            ax2.plot(df.index, df["Training ROC-AUC"], label="Train AUC")
            ax2.plot(df.index, df["Validation ROC-AUC"], label="Val AUC")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("AUC")
            ax2.set_title("Training vs Validation ROC-AUC")
            ax2.legend()

            # Save with epoch in filename
            fig_path = f"outputs/{model.name}/plots/loss_auc_curves_epoch{epoch_num:04d}.png"
            fig.savefig(fig_path)
            plt.close(fig)

            # Also (optionally) save/update a "latest" plot
            latest_path = f"outputs/{model.name}/plots/loss_auc_curves_lr_{LR}_wd_{WD}.png"
            os.replace(fig_path, latest_path)
            print(f"Saved loss/AUC curves to {latest_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
