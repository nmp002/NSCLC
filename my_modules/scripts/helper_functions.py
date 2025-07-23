import os
import random

import torchvision.transforms.v2 as transforms
from PIL import Image
import torch
import numpy as np


# Function to load TIFFs
def load_tiff(load_fns_and_img_paths):
    toTensor = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
    x = toTensor(Image.open(load_fns_and_img_paths[1])).to(torch.float32)
    return x


# Function to load ASCs
def load_asc(load_fns_and_img_paths):
    x = torch.from_numpy(np.genfromtxt(load_fns_and_img_paths[1])).to(torch.float32).unsqueeze(0)
    return x


# Function to simplify getting mean lifetime from raw parameters
def load_weighted_average(load_fns_and_img_paths):
    x = torch.tensor([], dtype=torch.float32)
    for fp in load_fns_and_img_paths[1]:
        x = torch.cat((x, torch.unsqueeze(fp[0](fp), dim=0)), dim=0)
    x = (x[0] * x[1] + x[2] * x[3]) / (x[0] + x[2])
    x[x < 0] = 0
    return x


# Function to simplify getting mean lifetime from raw parameters
def load_bound_fraction(load_fns_and_img_paths):
    x = torch.tensor([], dtype=torch.float32)
    for fp in load_fns_and_img_paths[1]:
        x = torch.cat((x, torch.unsqueeze(fp[0](fp), dim=0)), dim=0)
    x = x[1] / (x[0] + x[1])
    x[x < 0] = 0
    x[x > 1] = 1
    return x


def load_intensity(load_fns_and_img_paths):
    x = torch.tensor([], dtype=torch.float32)
    for fp in load_fns_and_img_paths[1]:
        x = torch.cat((x, torch.unsqueeze(fp[0](fp), dim=0)), dim=0)
    x = torch.nanmean(x, dim=0)
    x[x < 0] = 0
    return x


def convert_mp_to_torch(mp_array, shape,
                        device=(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))):
    np_array = np.ctypeslib.as_array(mp_array.get_obj())
    np_array = np.reshape(np_array, shape)
    torch_array = torch.from_numpy(np_array).to(device)
    return torch_array


def patient_wise_train_test_splitter(data, n=3):
    # Split data by patients, ensuring n patients per class in test set
    # Sample patients randomly
    subsampler = torch.utils.data.sampler.SubsetRandomSampler(range(data.patient_count))
    idx = [i for i in subsampler]

    # Get the image indices for all patients as nested lists
    patient_subsets = [data.get_patient_subset(i) for i in idx]

    # Find and remove any patients with no image indices
    idx_for_removal = []
    for i, subset in enumerate(patient_subsets):
        if len(subset) == 0:
            idx_for_removal.append(idx[i])
    for ix in idx_for_removal:
        idx.remove(ix)

    # Get labels for all remaining patients
    labels = [data.get_patient_label(i).item() for i in idx]

    # Separate 0 and 1 labels (still shuffled)
    shuffled_zeros = [i for i, l in zip(idx, labels) if l == 0]
    shuffled_ones = [i for i, l in zip(idx, labels) if l == 1]

    # Assign first three of each class to test set and expand to full image indices
    test_subjects = shuffled_zeros[:n] + shuffled_ones[:n]
    test_subs = [data.get_patient_subset(i) for i in test_subjects]
    test_indices = [i for sub in test_subs for i in sub]

    # Assign remaining patients to train set
    train_subjects = shuffled_zeros[n:] + shuffled_ones[n:]
    train_subs = [data.get_patient_subset(k) for k in train_subjects]
    train_indices = [i for sub in train_subs for i in sub]

    # Shuffle and subset
    random.shuffle(test_indices)
    random.shuffle(train_indices)
    test_set = torch.utils.data.Subset(data, test_indices)
    train_set = torch.utils.data.Subset(data, train_indices)
    return train_set, test_set


def subdivide_list(list_to_subdivide, num_parts):
    length = len(list_to_subdivide)
    quotient, remainder = divmod(length, num_parts)

    subdivisions = []
    start = 0
    for i in range(num_parts):
        end = start + quotient + (1 if i < remainder else 0)
        subdivisions.append(list_to_subdivide[start:end])
        start = end

    return subdivisions


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")