# stageII_fixed_split_stageI_train_only_4planed_2500epochs.py
# Train 5-planed ResNet18 variant on fixed 17 Stage II patients,
# with held-out 8 Stage II patients as validation set.
# Inputs: ['fad', 'nadh', 'shg', 'orr'] (4 planes).
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
from my_modules.scripts.dataset import NSCLCDataset

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
FAST_TEST = False
TOTAL_EPOCHS = 2500
SAVE_INTERVAL = 250

TRAIN_PTS = [26, 22, 28, 24, 33, 17, 31, 25, 27, 21, 13, 16, 35, 19, 20, 15, 32]
TEST_PTS_STAGEII = [23, 18, 34, 37, 36, 14, 29, 30]

LR = 1e-7
WD = 0.005


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    set_seed(42)
    random.seed(42)
    np.random.seed(42)

    try:
        mp.set_start_method('forkserver', force=True)
    except RuntimeError:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    modes = ['fad', 'nadh', 'shg', 'orr']

    train_data = NSCLCDataset('NSCLC_Data_for_ML', modes,
                              device=torch.device('cpu'),
                              label='Metastases', mask_on=True)

    eval_data = NSCLCDataset('NSCLC_Data_for_ML', modes,
                             device=torch.device('cpu'),
                             label='Metastases', mask_on=True)

    if FAST_TEST:
        train_data.augmented = False
        eval_data.augmented = False
        train_data.normalize_method = 'preset'
        eval_data.normalize_method = 'preset'
        train_data.transforms = None
        eval_data.transforms = None
        train_data.to(device)
        eval_data.to(device)
        total_epochs = 3
        batch_size = 8
    else:
        train_data.augment()
        train_data.normalize_method = 'preset'
        train_data.to(device)
        train_data.transforms = tvt.Compose([
            tvt.RandomVerticalFlip(p=0.25),
            tvt.RandomHorizontalFlip(p=0.25),
            tvt.RandomRotation(degrees=(-180, 180))
        ])

        eval_data.normalize_method = 'preset'
        eval_data.to(device)
        total_epochs = TOTAL_EPOCHS
        batch_size = 64

    # Flatten image indices
    train_img_idx = [im for p in TRAIN_PTS for im in train_data.get_patient_subset(p)]
    random.shuffle(train_img_idx)

    val_img_idx_stageII = [im for p in TEST_PTS_STAGEII for im in eval_data.get_patient_subset(p)]
    random.shuffle(val_img_idx_stageII)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    train_set = torch.utils.data.Subset(train_data, train_img_idx)
    val_set_stageII = torch.utils.data.Subset(eval_data, val_img_idx_stageII)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader_stageII = torch.utils.data.DataLoader(val_set_stageII, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # Model (modified to return logits)
    # ------------------------------------------------------------------
    model = ResNet18NPlaned(train_data.shape, start_width=64, n_classes=1)

    ### CHANGED: Ensure model outputs logits
    # (Modify your ResNet model separately if it has a sigmoid inside.)
    # Must REMOVE any torch.sigmoid() at the end of forward()

    model.to(device)

    # ------------------------------------------------------------------
    # Training setup
    # ------------------------------------------------------------------

    ### CHANGED: Use BCEWithLogitsLoss (logits expected)
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    # Output directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs(f'outputs/{model.name}/plots', exist_ok=True)
    os.makedirs(f'outputs/{model.name}/models', exist_ok=True)

    train_loss, train_auc = [], []
    val_loss, val_auc = [], []

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for ep in range(total_epochs):
        print(f'\nEpoch {ep + 1}/{total_epochs}')

        # ---- Training ----
        model.train()
        epoch_train_loss = 0.0
        train_logits = []
        train_targets_list = []

        for x, target in train_loader:
            x, target = x.to(device), target.to(device)

            logits = model(x)                        # (B,1) logits
            loss = loss_fn(logits, target.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_logits.append(logits.detach().cpu())
            train_targets_list.append(target.detach().cpu())

        epoch_train_loss /= max(len(train_set), 1)
        train_loss.append(epoch_train_loss)

        # AUC computed on *sigmoid(logits)*
        train_logits = torch.cat(train_logits).numpy()
        train_targets_array = torch.cat(train_targets_list).numpy()

        ### CHANGED: Apply sigmoid for AUC
        train_probs = 1 / (1 + np.exp(-train_logits))

        try:
            epoch_train_auc = roc_auc_score(train_targets_array, train_probs)
        except Exception:
            epoch_train_auc = 0.0

        train_auc.append(epoch_train_auc)
        print(f">>> Train Loss={epoch_train_loss:.4f}, Train AUC={epoch_train_auc:.4f}")

        # ---- Validation ----
        model.eval()
        epoch_val_loss = 0.0
        val_logits = []
        val_targets_list = []

        with torch.no_grad():
            for x_val, target_val in val_loader_stageII:
                x_val, target_val = x_val.to(device), target_val.to(device)

                logits_val = model(x_val)
                loss_val = loss_fn(logits_val, target_val.unsqueeze(1))

                epoch_val_loss += loss_val.item()
                val_logits.append(logits_val.detach().cpu())
                val_targets_list.append(target_val.detach().cpu())

        epoch_val_loss /= max(len(val_set_stageII), 1)
        val_loss.append(epoch_val_loss)

        val_logits = torch.cat(val_logits).numpy()
        val_targets_array = torch.cat(val_targets_list).numpy()

        ### CHANGED: Sigmoid during evaluation
        val_probs = 1 / (1 + np.exp(-val_logits))

        try:
            epoch_val_auc = roc_auc_score(val_targets_array, val_probs)
        except Exception:
            epoch_val_auc = 0.0

        val_auc.append(epoch_val_auc)
        print(f">>> Val Loss={epoch_val_loss:.4f}, Val AUC={epoch_val_auc:.4f}")

        # ---- Save checkpoints ----
        epoch_num = ep + 1
        if epoch_num % SAVE_INTERVAL == 0 or epoch_num == total_epochs:

            model_path = f"outputs/{model.name}/models/{model.name}_lr_{LR}_wd_{WD}_epoch{epoch_num}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Saved model checkpoint: {model_path}")

            df = pd.DataFrame({
                "Training Loss": train_loss,
                "Validation Loss": val_loss,
                "Training ROC-AUC": train_auc,
                "Validation ROC-AUC": val_auc
            }, index=range(1, len(train_loss) + 1))

            df.to_csv(f"outputs/{model.name}/tabular_train_val_lr_{LR}_wd_{WD}.csv", index_label="Epoch")

            # Plot curves
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
            plt.suptitle(f"{model.name} (up to epoch {epoch_num})")

            ax1.plot(df.index, df["Training Loss"], label="Train Loss")
            ax1.plot(df.index, df["Validation Loss"], label="Val Loss")
            ax1.set_title("Training vs Validation Loss")
            ax1.legend()

            ax2.plot(df.index, df["Training ROC-AUC"], label="Train AUC")
            ax2.plot(df.index, df["Validation ROC-AUC"], label="Val AUC")
            ax2.set_title("Training vs Validation ROC-AUC")
            ax2.legend()

            fig_path = f"outputs/{model.name}/plots/loss_auc_curves_epoch{epoch_num:04d}.png"
            fig.savefig(fig_path)
            plt.close(fig)

            # Update latest
            latest_path = f"outputs/{model.name}/plots/loss_auc_curves_lr_{LR}_wd_{WD}.png"
            os.replace(fig_path, latest_path)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
