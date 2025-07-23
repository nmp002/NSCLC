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

    # Random split datasets
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

    # Get labels for all remaining patients
    labels = [train_data.get_patient_label(i).item() for i in idx]
    image_counts = [0, 0]
    for i, label in zip(idx, labels):
        image_counts[int(label)] += len(train_data.get_patient_subset(i))

    # Separate 0 and 1 labels (still shuffled)
    shuffled_zeros = [i for i, l in zip(idx, labels) if l == 0]
    shuffled_ones = [i for i, l in zip(idx, labels) if l == 1]
    print(f'Total non-metastatic patients: {len(shuffled_ones)} with {image_counts[1]} images')
    print(f'Total metastatic patients: {len(shuffled_zeros)} with {image_counts[0]} images')

    # Split train, eval, and test sets
    train_pts = shuffled_zeros[3:-1] + shuffled_ones[3:-1]
    train_idx = [train_data.get_patient_subset(i) for i in train_pts]
    train_idx = [im for i in train_idx for im in i]
    random.shuffle(train_idx)
    train_image_counts = [0, 0]
    for idx in train_pts:
        label = train_data.get_patient_label(idx)
        train_image_counts[int(label)] += len(train_data.get_patient_subset(idx))

    eval_pts = shuffled_zeros[0:3] + shuffled_ones[0:3]
    eval_idx = [eval_test_data.get_patient_subset(i) for i in eval_pts]
    eval_idx = [im for i in eval_idx for im in i]
    random.shuffle(eval_idx)
    eval_image_counts = [0, 0]
    for idx in eval_pts:
        label = eval_test_data.get_patient_label(idx)
        eval_image_counts[int(label)] += len(eval_test_data.get_patient_subset(idx))

    test_pts = [shuffled_zeros[-1], shuffled_ones[-1]]
    test_idx = [eval_test_data.get_patient_subset(i) for i in test_pts]
    test_idx = [im for i in test_idx for im in i]
    random.shuffle(test_idx)
    test_image_counts = [0, 0]
    for idx in test_pts:
        label = eval_test_data.get_patient_label(idx)
        test_image_counts[int(label)] += len(eval_test_data.get_patient_subset(idx))

    comb_pts = eval_pts + test_pts
    comb_idx = [eval_test_data.get_patient_subset(i) for i in comb_pts]
    comb_idx = [im for i in comb_idx for im in i]
    random.shuffle(comb_idx)
    comb_image_counts = [0, 0]
    for idx in comb_pts:
        label = eval_test_data.get_patient_label(idx)
        comb_image_counts[int(label)] += len(eval_test_data.get_patient_subset(idx))

    print(f'Training set\n'
          f'____________\n'
          f'Non-metastatic: {len(shuffled_ones[3:-1])} with {train_image_counts[1]} images.\n'
          f'Metastatic: {len(shuffled_zeros[3:-1])} with {train_image_counts[0]} images.\n'
          f'Total: {len(train_pts)} Patients with {len(train_idx)} images.\n')

    print(f'Evaluation set\n'
          f'______________\n'
          f'Non-metastatic: {len(shuffled_ones[0:3])} with {eval_image_counts[1]} images.\n'
          f'Metastatic: {len(shuffled_zeros[0:3])} with {eval_image_counts[0]} images.\n'
          f'Total: {len(eval_pts)} Patients with {len(eval_idx)} images.\n')

    print(f'Testing set\n'
          f'____________\n'
          f'Non-metastatic: {1} with {test_image_counts[1]} images.\n'
          f'Metastatic: {1} with {test_image_counts[0]} images.\n'
          f'Total: {len(test_pts)} Patients with {len(test_idx)} images.\n')

    print(f'Training patients: {train_pts}.\nEvaluation patients: {eval_pts}.\nTest patients: {test_pts}.\n')

    # Create dataloaders for fold
    batch_size = 64
    train_set = torch.utils.data.Subset(train_data, train_idx)
    eval_set = torch.utils.data.Subset(eval_test_data, eval_idx)
    test_set = torch.utils.data.Subset(eval_test_data, test_idx)
    comb_set = torch.utils.data.Subset(eval_test_data, comb_idx)

    # Create loaders
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
    # ResNet18
    models = [ResNet18NPlaned(train_data.shape, start_width=64, n_classes=1)]

    # InceptionResNetV2 Feature Extractor (BigCoMET)
    # feature_extractor = AdaptedInputInceptionResNetV2(train_data.shape, num_classes=1000, pretrained=False)
    # classifier = CometClassifierWithBinaryOutput
    # models.append(FeatureExtractorToClassifier(train_data.shape,
    #                                            feature_extractor=feature_extractor,
    #                                            classifier=classifier, layer='inceptionresnetv2.conv2d_7b'))

    # Xception Feature Extractor
    # feature_extractor = AdaptedInputXception(train_data.shape, num_classes=1000, pretrained=False)
    # classifier = torch.nn.Sequential(torch.nn.Linear(2048, 1), torch.nn.Sigmoid())
    # models.append(FeatureExtractorToClassifier(train_data.shape,
    #                                            feature_extractor=feature_extractor,
    #                                            classifier=classifier, layer='xception.conv4'))

    # Basic CNNs
    models[len(models):] = [CNNet(train_data.shape),
                            RegularizedCNNet(train_data.shape),
                            ParallelCNNet(train_data.shape),
                            RegularizedParallelCNNet(train_data.shape)]

    # Put all models on GPU if available
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
    # Prep results file
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

    # Testing
    headers = ['Best Test', 'Best Eval & Test']
    data = [[] for _ in range(len(models))]
    for i, model in enumerate(models):
        # best eval model
        # on test set
        print(f'\n>>> {model.name} at best evaluated on test set...')
        model.load_state_dict(torch.load(f'outputs/{model.name}/models/Best {model.name}.pth'))
        scores, fig = score_model(model, test_loader, print_results=True, make_plot=True, threshold_type='roc')
        fig.savefig(f'outputs/{model.name}/plots/best_eval_{model.name}_on_test_plots.png')
        plt.close(fig)
        with open(f'outputs/{model.name}/results.txt', 'a') as f:
            f.write(f'\n>>> {model.name} at best evaluated on test set...')
            for key, item in scores.items():
                if 'Confusion' not in key:
                    f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')
            f.write('_____________________________________________________\n')
        data[i].append(scores['ROC-AUC'])

        # on eval-test set
        print(f'\n>>> {model.name} at best evaluated on eval and test sets...')
        scores, fig = score_model(model, comb_loader, print_results=True, make_plot=True, threshold_type='roc')
        fig.savefig(f'outputs/{model.name}/plots/best_eval_{model.name}_on_eval-test_plots.png')
        plt.close(fig)
        with open(f'outputs/{model.name}/results.txt', 'a') as f:
            f.write(f'\n>>> {model.name} at best evaluated on eval and test sets...')
            for key, item in scores.items():
                if 'Confusion' not in key:
                    f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')
            f.write('_____________________________________________________\n')
        data[i].append(scores['ROC-AUC'])

        # At checkpoint epochs
        for ep in epochs:
            headers.append(f'{ep} Epoch Test') if f'{ep} Epoch Test' not in headers else None
            # best eval model
            # on test set
            print(f'\n>>> {model.name} at {ep} on test set...')
            model.load_state_dict(torch.load(f'outputs/{model.name}/models/Epochs {ep} {model.name}.pth'))
            scores, fig = score_model(model, test_loader, print_results=True, make_plot=True, threshold_type='roc')
            fig.savefig(f'outputs/{model.name}/plots/best_eval_{model.name}_on_test_plots.png')
            plt.close(fig)
            with open(f'outputs/{model.name}/results.txt', 'a') as f:
                f.write(f'\n>>> {model.name} at {ep} epochs on test set...')
                for key, item in scores.items():
                    if 'Confusion' not in key:
                        f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')
                f.write('_____________________________________________________\n')
            data[i].append(scores['ROC-AUC'])

            # on eval-test set
            headers.append(f'{ep} Epoch Eval & Test') if f'{ep} Epoch Eval & Test' not in headers else None
            print(f'\n>>> {model.name} at {ep} on eval and test sets...')
            scores, fig = score_model(model, comb_loader, print_results=True, make_plot=True, threshold_type='roc')
            fig.savefig(f'outputs/{model.name}/plots/best_eval_{model.name}_on_eval-test_plots.png')
            plt.close(fig)
            with open(f'outputs/{model.name}/results.txt', 'a') as f:
                f.write(f'\n>>> {model.name} at {ep} epochs on eval and test sets...')
                for key, item in scores.items():
                    if 'Confusion' not in key:
                        f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')
                f.write('_____________________________________________________\n')
            data[i].append(scores['ROC-AUC'])
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
