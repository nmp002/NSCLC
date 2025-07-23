import warnings

import torchvision.transforms.v2 as t
import os

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from .helper_functions import (load_tiff, load_asc, load_intensity, load_weighted_average, load_bound_fraction,
                               convert_mp_to_torch)
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import re
import glob
import multiprocessing as mp
import ctypes
from torch.utils.data import Dataset


class NSCLCDataset(Dataset):
    """
    NSCLC dataset class to load NSCLC images from root dir (arg). When __getitem__ is used, the dataset will return a
    tuple with the fov stack of modes (ordered by the input, or default order with 'all') at index (arg) and the binary
    label of the sample, or the mask if 'mask' is set for the label. IN the binary classes, the label will be 0 for
    non-responders or positive metastases and 1 otherwise. As a shorthand, a positive class (1) is the positive outcome.

    Image modes are specified at creation and can include any typical MPM image modes available from the dataset and
    derived modes (bound fraction and mean lifetime).
TODO: Update doc
    Attributes:
        - root (string): Root directory of NSCLC dataset.
        - mode (list): Ordered list of image modes as they will be returned.
        - label (string): Name of feature to be returned as data label.
        - stack_height (int): Height of image stack, equivalent to number of modes
        - image_dims (tuple): Tuple of image dimensions.
        - scalars (arr): Max value of each mode used in normalization
        - name (string): Name of the dataset dependent on data parameters.
        - shape (tuple): Shape of individual image stack
        - reset_cache (callable): Function to reset cached data in shared caches
        - dist_transform (callable): Applies distribution transform to image stack
        - dist_transformed (bool): Returns whether image stack is histograms
        - augment (callable): Applies image augmentation to dataset using FiveCrop augmentation
        - augmented (bool): Returns whether dataset is augmented
        - show_random (callable): Shows 5 random samples from dataset
        - device (str or torch.device): Device type or device ID
        - to (callable): Move any currently cached items to DEVICE from input argument in call and return all future
            items on DEVICE.
    """

    # region Main Dataset Methods -- init, len, getitem (with helper methods included)
    def __init__(self, root, mode, xl_file=None, label=None, mask_on=True, transforms=None, use_atlas=False,
                 use_patches=True, patch_size=(512, 512), use_cache=True, pool_patients=False, remove_empties=True,
                 device=(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))):
        super().__init__()
        self.transforms = transforms
        self.root = root
        self.device = device
        self.label = label
        self._use_atlas = use_atlas
        self.mode = mode
        self.mask_on = False if self._use_atlas else mask_on
        self.use_cache = use_cache
        self.use_patches = use_patches if self._use_atlas else False
        self._patch_size = patch_size
        self.stack_height = len(self.mode)
        self.image_dims = None
        self.scalars = {}

        # Set attribute and property defaults
        self.pool_patients = pool_patients
        self._augment_patients = False
        self.saturate = False
        self._augmented = False
        self.filter_bad_data = True if self._use_atlas else False
        self.dist_transformed = False
        self.psuedo_rgb = False
        self.rgb_squeeze = False if self.stack_height > 1 else True
        self._name = 'nsclc_'
        self._shape = None
        self._normalize_method = None
        self._nbins = 25

        # Init placeholder cache arrays
        self.index_cache = None
        self.in_bad_index_cache = None
        self.shared_x = None
        self.shared_y = None

        # Find and load features spreadsheet (or load directly if path provided)
        if xl_file is None:
            xl_file = glob.glob(self.root + os.sep + '*.xlsx')
            if not xl_file:
                raise Exception(
                    'Features file not found,'
                    ' input path manually at dataset initialization using xl_file=<path_to_file>.')
            self.xl_file = xl_file[0]
        else:
            self.xl_file = xl_file
        self.features = pd.read_excel(self.xl_file, header=[0, 1])
        self.total_patient_count = len(self.features)
        """
        - atlases_by_sample and fovs_by_subject are lists of lists. The inner lists correspond to the images within a 
        given sample or slide and the outer lists match the index of the slide to the features. In other words, the 
        index of a given slide in the features file will also match a list of all images from that slide in the list of 
        lists.
        - all_atlases and all_fovs maintain the order, but un-nest the lists so they can easily be indexed into 
        to actually use the paths to get items.  
        - atlas_mode_dict and fov_mode_dict are lists of dicts. Each dict matches the index of the all_... lists, but 
        also includes functions and paths for individual modes. They are of the form:
            ..._mode_dict[<image_index_from_all_list>] = {<mode name>: [<specific load function for mode's file type>,
                                                                        <path for specific mode of indexed image and>]
                                                           ...}
           In the case of atlases, this mode dict is trivial because there is only one load function (since only 'orr' 
           is available), but it is still produced to easily match loading at __getitem__ for both data types.
           
           Whichever data type will be used is added as simply img_... variables to easily pass to __getitem__ 
           regardless of datatype. After __init__ this should make the behavior the same (with the exception of cropping
           and __len__ for atlases).      
        """
        # Prepare a list of images from data dir matched to slide names from the features excel file
        if self._use_atlas:
            # Track number of individual 512x512 patches available within each atlas
            self.atlas_patch_dims = []

            # Track running total of patches through each index for sub-indexing later
            total_patches_through_index = 0
            self.atlas_sub_index_map = [0]

            # Nested list of atlas locations nested by sample (for label indexing)
            self.atlases_by_sample = []

            # A "mode_dict" that looks the same as the fov_mode_dict (though this one is trivial)
            self.atlas_mode_dict = []
            for subject in self.features['ID']['Sample']:
                sample_dir = os.path.join(self.root, 'Atlas_Images', subject)
                self.atlases_by_sample.append([])
                for trunk, dirs, files in os.walk(sample_dir):
                    for f in files:
                        if 'rawredoxmap.tiff' == f.lower():
                            im_path = os.path.join(trunk, f)
                            self.atlases_by_sample[-1].append(trunk)
                            self.atlas_mode_dict.append({'orr': [load_tiff, im_path]})
                            with Image.open(im_path) as im:
                                width, height = im.size
                                rm_width, rm_height = width % self._patch_size[1], height % self._patch_size[0]
                                width, height = width - rm_width, height - rm_height  # "crop" to be a perfect fit
                                self.atlas_patch_dims.append(
                                    (height / self._patch_size[0], width / self._patch_size[1]))
                                total_patches_through_index += np.prod(self.atlas_patch_dims[-1])
                                self.atlas_sub_index_map.append(total_patches_through_index)

            # Un-nest list (still ordered, but now easily indexable)
            self.all_atlases = [atlas for sample_atlases in self.atlases_by_sample for atlas in sample_atlases]
        else:
            self.fovs_by_subject = []
            for subject in self.features['ID']['Subject']:
                self.fovs_by_subject.append([])
                # Walk the root until files are hit
                for rt, dr, f in os.walk(f'{self.root}{os.sep}{subject}{os.sep}'):
                    # When files are hit, add that dir to the nested list for the subject
                    if f:
                        self.fovs_by_subject[-1].append(rt)
            self.all_fovs = [fov for slide_fovs in self.fovs_by_subject for fov in
                             slide_fovs]  # This will give a list of all fovs (still ordered, but now not nested,
            # making it simple for indexing in __getitem__)

            # Master dicts for easy sorting of FOVs by file names and easy addition of appropriate load function based
            # on type
            # Define loading functions for different image types
            load_fn = {'tiff': load_tiff,
                       'asc': load_asc,
                       'int': load_intensity,
                       'weighted_average': load_weighted_average,
                       'ratio': load_bound_fraction}

            # Define a mode dict that matches appropriate load functions and filename regex to mode
            self.mode_dict = {'mask': [load_fn['tiff'], r'mask\.(tiff|TIFF)'],
                              'nadh': [load_fn['tiff'], r'nadh\.(tiff|TIFF)'],
                              'fad': [load_fn['tiff'], r'fad\.(tiff|TIFF)'],
                              'shg': [load_fn['tiff'], r'shg\.(tiff|TIFF)'],
                              'orr': [load_fn['tiff'], r'orr\.(tiff|TIFF)'],
                              'g': [load_fn['asc'], r'(G|g)\.(asc|ASC)'],
                              's': [load_fn['asc'], r'(S|s)\.(asc|ASC)'],
                              'photons': [load_fn['asc'], r'photons\.(asc|ASC)'],
                              'tau1': [load_fn['asc'], r't1\.(asc|ASC)'],
                              'tau2': [load_fn['asc'], r't2\.(asc|ASC)'],
                              'alpha1': [load_fn['asc'], r'a1\.(asc|ASC)'],
                              'alpha2': [load_fn['asc'], r'a2\.(asc|ASC)']}
            # Compile regexes
            self.mode_dict = {key: [item[0], re.compile(rf'.*?[/\\]{item[1]}')] for key, item in
                              self.mode_dict.items()}

            # Make an indexed FOV-LUT dict list of loaders and files
            self.fov_mode_dict = [{} for _ in range(len(self.all_fovs))]
            # Iterate through all FOVs
            for index, fov in enumerate(self.all_fovs):
                # Iterate through all base modes (from mode_dict)
                for mode, (load_fn, file_pattern) in self.mode_dict.items():
                    matched = []
                    # Iterate through the current FOV tree
                    for trunk, dirs, files in os.walk(fov):
                        # Check if any file in the tree matches the pattern for the mode from the base LUT (mode_dict)
                        for file in files:
                            matched.append(re.match(file_pattern, os.path.join(trunk, file)))
                    # If exactly one file matched, then add it to the FOV-LUT dict
                    if sum(path_str is not None for path_str in matched) == 1:
                        for match in matched:
                            if match:
                                self.fov_mode_dict[index][mode] = [load_fn, match.string]
                    # Else, add <None> for later removal and move on to next FOV
                    else:
                        self.fov_mode_dict[index][mode] = [fov, None]

                # Add derived modes
                self.fov_mode_dict[index]['boundfraction'] = [load_bound_fraction, [self.fov_mode_dict[index]['alpha1'],
                                                                                    self.fov_mode_dict[index]['alpha2']
                                                                                    ]]
                self.fov_mode_dict[index]['taumean'] = [load_weighted_average,
                                                        [self.fov_mode_dict[index]['alpha1'],
                                                         self.fov_mode_dict[index]['tau1'],
                                                         self.fov_mode_dict[index]['alpha2'],
                                                         self.fov_mode_dict[index]['tau2']
                                                         ]]
                self.fov_mode_dict[index]['intensity'] = [load_intensity, [self.fov_mode_dict[index]['nadh'],
                                                                           self.fov_mode_dict[index]['fad']
                                                                           ]]
            # Remove items that are missing a called mode
            # Note the [:] makes a copy of the list so indices don't change on removal
            for ii, fov_lut in enumerate(self.fov_mode_dict[:]):
                for mode in self.mode:
                    match mode.lower():
                        case 'taumean':
                            if not all([fov_lut['alpha1'][1], fov_lut['tau1'][1],
                                        fov_lut['alpha2'][1], fov_lut['tau2'][1]]):
                                self.all_fovs.remove(fov_lut['alpha1'][0])
                                self.fov_mode_dict.remove(fov_lut)
                                print(f'1removed {fov_lut[mode]} due to {mode}')
                                break
                        case 'boundfraction':
                            if not all([fov_lut['alpha1'][1], fov_lut['alpha2'][1]]):
                                self.all_fovs.remove(fov_lut['alpha1'][0])
                                self.fov_mode_dict.remove(fov_lut)
                                print(f'2removed {fov_lut[mode]} due to {mode}')
                                break
                        case 'intensity':
                            if not all([fov_lut['fad'][1], fov_lut['nadh'][1]]):
                                print('not')
                                self.all_fovs.remove(fov_lut['fad'][0])
                                self.fov_mode_dict.remove(fov_lut)
                                print(f'3removed {fov_lut[mode]} due to {mode}')
                        case _:
                            if fov_lut[mode][1] is None:
                                self.all_fovs.remove(fov_lut[mode][0])
                                self.fov_mode_dict.remove(fov_lut)
                                print(f'removed {fov_lut[mode]} due to {mode}')
                                break

        self.patient_count = self.total_patient_count

        # Remove empty patients entirely
        if remove_empties:
            self.remove_empty()

    def remove_empty(self):
        for_removal = []
        for pt in range(self.total_patient_count):
            if len(self.get_patient_subset(pt)) == 0:
                self.patient_count -= 1
                for_removal.append(pt)

        for pt in for_removal:
            self.features.drop(pt, inplace=True)
            self.fovs_by_subject.remove([])

        # Reindex patients
        self.features.index = range(len(self.features))

    def __len__(self):
        if self._use_atlas:
            if self.use_patches:
                atlas_len = int(self.atlas_sub_index_map[-1])
            else:
                atlas_len = len(self.all_atlases)
            if self.augmented:
                return 5 * atlas_len
            else:
                return atlas_len
        elif self.pool_patients:
            if self.augment_patients:
                return 5 * self.patient_count
            else:
                return self.patient_count
        else:
            if self.augmented:
                return 5 * len(self.all_fovs)
            else:
                return len(self.all_fovs)

    def __parse_index__(self, index):
        # Parse the index
        # Get image path from index
        if self.augmented:
            base_index = int(index // 5)  # This will give us the index for the base img
            sub_index = index % 5  # This gives the index of the crop
        else:
            base_index = index
            sub_index = None
        if self._use_atlas:
            # Find where the index is <= than the number of patches
            if self.use_patches:
                path_index = 0
                while self.atlas_sub_index_map[path_index + 1] <= base_index:
                    path_index += 1
                patch_index = int(base_index - self.atlas_sub_index_map[path_index])
            else:
                path_index = base_index
                patch_index = None
            img_path = self.all_atlases[path_index]
            load_dict = self.atlas_mode_dict
            lists_of_paths = self.atlases_by_sample
        else:
            path_index = base_index
            img_path = self.all_fovs[path_index]
            load_dict = self.fov_mode_dict
            lists_of_paths = self.fovs_by_subject
            patch_index = None
        slide_idx = [img_path in paths for paths in lists_of_paths].index(True)

        # Return appropriate indices
        return slide_idx, path_index, sub_index, patch_index, load_dict

    def __getitem__(self, index, pool=None):
        # Handle pooling (recurse without pooling, so ust include it as arg)
        pool = self.pool_patients if pool is None else pool

        # Cat all the images for a patient together
        if pool:
            pool_idx = self.get_patient_subset(index)
            y = self.get_patient_label(index)
            if len(pool_idx) > 0:
                x_pool = torch.rand(*self.shape, len(pool_idx))
                for i, idx in enumerate(pool_idx):
                    x_pool[..., i], _ = self.__getitem__(idx, pool=False)
            else:
                x_pool = torch.tensor([])
            return x_pool, y

        slide_idx, path_index, sub_index, patch_index, load_dict = self.__parse_index__(index)

        # Check if this index has been previously deemed bad
        if self.in_bad_index_cache is not None and self.in_bad_index_cache[index]:
            print(f'skipping previous bad index {index}')
            pt_indices = self.get_patient_subset(slide_idx)
            while self.in_bad_index_cache[index]:
                index = pt_indices[(pt_indices.index(index) + 1) % len(pt_indices)]
            return self.__getitem__(index)

        # Check if indexed sample is in cache (by checking for index in index_cache)...
        # if it is, pull it from the cache;
        # Get base image and label from cache
        if self.index_cache is not None and path_index in self.index_cache:
            x = self.shared_x[path_index].clone()
            y = self.shared_y[path_index]
        # Load base image and label into cache
        else:
            # load the sample, and cache the sample (if cache is open)
            # region Load Data and Label

            # Load modes using load functions
            for ii, mode in enumerate(self.mode):
                # Pre-allocate on first pass
                mode_load = load_dict[path_index][mode][0](load_dict[path_index][mode]).to(self.device)
                if ii == 0:
                    self.image_dims = (self.stack_height,) + tuple(mode_load.size()[1:])
                    x = torch.empty(self.image_dims, dtype=torch.float32, device=self.device)
                x[ii] = mode_load

            # Get data label
            # Get index of nested list that contains the image path based on what slide the FOV is from. This index will
            # coincide with the index of the features file to get the label of the sample/slide the image is from.
            match self.label:
                case 'Response':
                    y = torch.tensor(1 if self.features['FOLLOWUP DATA']['Status (NR/R)'].iloc[slide_idx] == 'R'
                                     else 0, dtype=torch.float32, device=self.device)
                case 'Metastases':
                    y = torch.tensor(1 if self.features['FOLLOWUP DATA']['Status (Mets/NM)'].iloc[slide_idx] == 'NM'
                                     else 0, dtype=torch.float32, device=self.device)
                case 'Mask':
                    # Load mask (if on or label)
                    fov_mask = load_dict[path_index]['mask'][0](load_dict[path_index]['mask']).to(self.device)
                    fov_mask[fov_mask == 0] = float('nan')
                    y = fov_mask
                case None:
                    y = torch.tensor(-999999,  # Placeholder for NaN label
                                     dtype=torch.float32, device=self.device)
                case _:
                    raise Exception(
                        'An unrecognized label is in use. Update label attribute of dataset and try again.')

            # Add the loaded image data to the cache (open and add, if it's not open)
            if self.index_cache is None and self.use_cache:
                self._open_cache(x, y)
            if self.use_cache:
                self.shared_x[path_index] = x.clone()
                self.shared_y[path_index] = y.clone()
                self.index_cache[path_index] = path_index

        # Load mask (if on)
        if self.mask_on:
            fov_mask = load_dict[path_index]['mask'][0](load_dict[path_index]['mask']).to(self.device).squeeze()
            # Apply mask to appropriate channels (not SHG)
            for ch in range(len(x)):
                x[ch, fov_mask == 0] = float('nan') if self.mode[ch] != 'shg' else x[ch, fov_mask == 0]

        # Perform all data augmentations, transformations, etc. on base image
        # patch the atlas into individual images
        if self.use_patches:
            r = int(self.atlas_patch_dims[path_index][0] * self._patch_size[0])
            c = int(self.atlas_patch_dims[path_index][1] * self._patch_size[1])
            x = x[:, :r, :c]
            x = x.unfold(1, self._patch_size[0], self._patch_size[0])
            x = x.unfold(2, self._patch_size[1], self._patch_size[1])
            x = x.reshape(self.stack_height, -1, self._patch_size[0], self._patch_size[1])
            x = x[:, patch_index, :, :]

        # Crop and sub index if necessary
        if self.augmented:
            cropper = t.FiveCrop((int(x.shape[1] / 2), int(x.shape[2] / 2)))
            x = cropper(x)
            x = x[sub_index]

        # Cutoff at saturation thresholds
        if self.saturate:
            for ch, mode in enumerate(self.mode):
                sat_mask = x[ch] > self._preset_values[mode][1]
                x[ch, sat_mask] = self._preset_values[mode][1]

        # Scale by the scalars from normalization method
        if self.normalize_method is not None:
            for ch in range(self.stack_height):
                lower, upper = self.scalars[self.normalize_method][ch, :]
                x[ch] = ((x[ch] - lower) / (upper - lower))

        # Dynamically and recursively filter out bad data (namely, images with little or no signal) while maintain the
        # same patient index
        if self.filter_bad_data and self.is_bad_data(x):
            print(f'crap {index} added to bad cache')
            self.in_bad_index_cache[index] = True
            pt_indices = self.get_patient_subset(slide_idx)
            while self.in_bad_index_cache[index]:
                index = pt_indices[(pt_indices.index(index) + 1) % len(pt_indices)]
            return self.__getitem__(index)

        # Unsqueeze so "color" is dim 1 and expand to look like an RGB image
        # New image dims: (M, C, H, W), where M is the mode, C is the psuedo-color channel, H and W are height and width
        if self.psuedo_rgb:
            x = x.unsqueeze(1).expand(-1, 3, -1, -1)
            if self.rgb_squeeze:
                x = x.squeeze()

        # Apply distribution transform, if called
        if self.dist_transformed:
            x_dist = torch.empty((self.stack_height,) + (self._nbins,), dtype=torch.float32, device=self.device)
            for ch, mode in enumerate(x):
                x_dist[ch], _ = torch.histogram(mode.cpu(), bins=self._nbins, range=[0, 1], density=True)
            x = x_dist

        # Apply transforms that were input (if any)
        x = self.transforms(x) if self.transforms is not None else x

        return x, y
        # endregion

    def _open_cache(self, x, y):
        # Setup shared memory arrays (i.e. caches that are compatible with multiple workers)
        # Length of cache to hold all FULL images before any manipulation
        cache_len = len(self.all_atlases) if self._use_atlas else len(self.all_fovs)

        # Index cache to track what indices have been hit already. Negative initialization ensure no overlap with actual
        # cached indices
        index_cache_base = mp.Array(ctypes.c_int, cache_len * [-1])
        self.index_cache = convert_mp_to_torch(index_cache_base, (cache_len,), device=self.device)
        self._open_bad_index_cache()

        shared_x_base = cache_len * [mp.Array(ctypes.c_float, 0)]
        self.shared_x = [convert_mp_to_torch(x_base, 0, device=self.device) for x_base in shared_x_base]

        # Label-size determines cache size, so if no label is set, we will fill cache with -999999 at __getitem__
        match self.label:
            case 'Response' | 'Metastases' | None:
                shared_y_base = mp.Array(ctypes.c_float, cache_len * [-1])
                y_shape = ()
            case 'Mask':
                shared_y_base = mp.Array(ctypes.c_float, int(cache_len * np.prod(y.shape)))
                y_shape = tuple(y.shape)
            case _:
                raise Exception('An unrecognized label is in use that is blocking the cache from initializing. '
                                'Update label attribute of dataset and try again.')
        self.shared_y = convert_mp_to_torch(shared_y_base, (cache_len,) + y_shape, device=self.device)

        print('Cache opened.')

    def _open_bad_index_cache(self):
        # If filtering bad data, store a map of bad to good indices to avoid reprocessing the same bad samples. As we
        # find bad indices, we will set that index in the bad index cache to True. Then we can just check whether an
        # index was previously deemed bad by checking if the value at that index is True
        temp_pool = self.pool_patients
        self.pool_patients = False
        bad_index_base = mp.Array(ctypes.c_bool, len(self) * [False])
        self.in_bad_index_cache = convert_mp_to_torch(bad_index_base, (len(self),), device=self.device)
        self.pool_patients = temp_pool

    def to(self, device):
        # Move caches to device
        if self.index_cache is not None:
            self.index_cache = self.index_cache.to(device)
            for i, idx in enumerate(self.index_cache):
                self.index_cache[i] = idx.to(device)
            for i, x_cache in enumerate(self.shared_x):
                self.shared_x[i] = x_cache.to(device)
            self.shared_y = self.shared_y.to(device)
            for i, y in enumerate(self.shared_y):
                self.shared_y[i] = y.to(device)

        # Move any self-held tensors to device for ops compatibility
        if self.normalize_method is not None:
            self.scalars[self.normalize_method] = self.scalars[self.normalize_method].to(device)

        # Update device for future items
        self.device = device

    def get_patient_subset(self, pt_index):
        if self.augment_patients:
            subset_index = pt_index % 5
            pt_index = int(pt_index // 5)

        # Get actual index from input index (to handle negatives)
        pt_index = list(range(len(self.features)))[pt_index]
        if self._use_atlas:
            pt_id = self.features['ID'].at[pt_index, 'Sample']
            indices = [i for i, path_str in enumerate(self.all_atlases) if pt_id in path_str]
        else:
            pt_id = self.features['ID'].loc[pt_index, 'Subject']
            indices = [i for i, path_str in enumerate(self.all_fovs) if pt_id in path_str]

        # If using atlas patches, we have to determine how many patches come before this patient and add the number from
        # the patient from there
        if self.use_patches:
            im_before_current_pt = int(self.atlas_sub_index_map[indices[0]])
            number_of_ims_for_pt = int(sum([np.prod(self.atlas_patch_dims[i]) for i in indices]))
            indices = list(range(im_before_current_pt, im_before_current_pt + number_of_ims_for_pt))

        # If using augmenting, each base image results in 5 sequential daughter images
        if self.augmented:
            indices = [5 * idx + i for idx in indices for i in range(5)]
        # If augmenting patient, every 5th image of the patient is used
        if self.augment_patients:
            indices = [idx for i, idx in enumerate(indices) if i % 5 == subset_index]

        return indices

    def get_patient_label(self, pt_index):
        if self.augment_patients:
            pt_index = int(pt_index // 5)
        match self.label:
            case 'Response':
                y = torch.tensor(1 if self.features['FOLLOWUP DATA'].at[pt_index, 'Status (NR/R)'] == 'R' else 0,
                                 dtype=torch.float32, device=self.device)
            case 'Metastases':
                y = torch.tensor(1 if self.features['FOLLOWUP DATA'].at[pt_index, 'Status (Mets/NM)'] == 'NM' else 0,
                                 dtype=torch.float32, device=self.device)
            case 'Mask':
                # Load mask (if on or label)
                pass
            case None:
                y = torch.tensor(-999999,  # Placeholder for NaN label
                                 dtype=torch.float32, device=self.device)
            case _:
                raise Exception(
                    'An unrecognized label is in use. Update label attribute of dataset and try again.')
        return y

    def get_patient_ID(self, pt_index):
        if self.augment_patients:
            pt_index = int(pt_index // 5)
        if type(pt_index) is int:
            return self.features['ID'].at[pt_index, 'Subject']
        elif type(pt_index) is str:
            return list(self.features.index[self.features['ID']['Subject'] == pt_index])[0]

    def is_bad_data(self, x):
        return (torch.sum(x <= 0.1).item() + torch.sum(x >= 0.9).item()) > (0.60 * np.prod(x.shape))

    # endregion

    # region Properties
    # Name, shape, label, mode, mask
    # Use of property (instead of simple attribute) to define ensures automatic updates with latest data setup and/or
    # appropriate clearing of the cache

    # Name
    @property
    def name(self):
        self._name = (f"nsclc_{self.label}_{'+'.join(self.mode)}"
                      f'{"_Histogram" if self.dist_transformed else ""}'
                      f'{"_Augmented" if self.augmented else ""}'
                      f'{f"_NormalizedTo-{self.normalize_method}" if self.normalize_method else ""}'
                      f'{"_Masked" if self.mask_on else ""}')
        return self._name

    # Shape
    @property
    def shape(self):
        temp_filter = self.filter_bad_data
        self.filter_bad_data = False
        if self._use_atlas and not self.use_patches:
            warnings.warn('`shape` is ambiguous when using non-patched atlases.')
        self._shape = self.__getitem__(0, pool=False)[0].shape
        self.filter_bad_data = temp_filter
        return self._shape

    # Label
    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        assert label is not None, 'Label must be provided.'
        match label.lower():
            case 'response' | 'r':
                label = 'Response'
            case 'metastases' | 'mets' | 'm':
                label = 'Metastases'
            case 'mask':
                label = 'Mask'
                self.mask_on = False
            case _:
                raise Exception('Invalid data label entered. Allowed labels are "RESPONSE", "METASTASES", and "MASK".')
        if hasattr(self, '_label') and label != self.label:
            temp_filter = self.filter_bad_data
            self._open_cache(*self[0])
            self.filter_bad_data = temp_filter
        self._label = label

    # Modes
    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        # Set default/shortcut behavior
        mode = [mode] if type(mode) is not list else mode
        if mode is None:
            mode = ['orr'] if self._use_atlas else ['all']
        if mode == ['all']:
            mode = ['intensity', 'orr', 'shg', 'photons', 'taumean', 'boundfraction']
        if self._use_atlas and mode[:] != 'orr':
            warnings.warn(f"{mode} is incompatible with atlases. Mode is being reset to 'orr'.")
            mode = ['orr']
        # If this is not the __init__ run
        if hasattr(self, '_mode') and mode != self.mode:
            # A mode update is essentially a new dataset, so we will re-init it, but we want to make all the other
            # aspects match, so we check and store them first. Once the dataset is reinitialized, we can reset the
            # other properties.
            temp_aug = self.augmented
            temp_norm = self.normalize_method
            temp_dist = self.dist_transformed

            # Hard set new mode then re-init with new mode (hard set prevents recursion)
            self._mode = mode

            # Re-init with new modes
            self.__init__(self.root, mode, xl_file=self.xl_file, label=self.label, mask_on=self.mask_on,
                          transforms=self.transforms, device=self.device)

            # Reset properties
            self.augmented = temp_aug
            self.normalize_method = temp_norm
            self.dist_transformed = temp_dist
        # If this is the __init__ run
        else:
            self._mode = mode

    # endregion

    # region Transforms and Augmentations
    def dist_transform(self, nbins=25):
        # If it is already transformed to the same bin number, leave it alone
        if self.dist_transformed and nbins == self._nbins:
            pass
        # If something is changed, reset and update
        else:
            # self.reset_cache()
            self._nbins = nbins
            if not self.normalize_method:
                print(
                    'Normalization to presets is automatically applied for the distribution transform.\n     '
                    'This can be manually overwritten by setting the NORMALIZED attribute to False after transforming. '
                    "To use max normalization, use normalize_method='minmax' before transforming.")
                self.normalize_method = 'preset'
        self.dist_transformed = True

    def transform_to_psuedo_rgb(self, rgb_squeeze=None):
        if rgb_squeeze is None:
            self.rgb_squeeze = False if self.stack_height > 1 else True
        self.psuedo_rgb = True

    def cutoff_saturation(self):
        self.saturate = True

    #region Augmentation
    def augment(self):
        self.augmented = True

    @property
    def augmented(self):
        return self._augmented

    @augmented.setter
    def augmented(self, augmented):
        # Updates method and the size of bad index cache because it holds a single value for each possible index, not
        # base image wise like the others
        if augmented is not self.augmented:
            self._augmented = augmented
            self._open_bad_index_cache()

    @property
    def augment_patients(self):
        return self._augment_patients

    @augment_patients.setter
    def augment_patients(self, augment_patients):
        self.augmented = augment_patients if augment_patients else self.augmented
        self._augment_patients = augment_patients

    #endregion

    # region Normalization
    @property
    def normalize_method(self):
        return self._normalize_method

    @normalize_method.setter
    def normalize_method(self, method):
        # Set _normalized_method so images will be scaled when retrieved
        match method:
            case 'minmax' | 'max':
                self._normalize_method = 'minmax'
            case 'preset':
                self._normalize_method = 'preset'
            case None | False:
                self._normalize_method = None
            case _:
                raise Exception(f"Invalid normalization method: {method}. Use 'minmax' or 'preset'.")

        # Determine appropriate scalars if a normalization method was set
        if self.normalize_method not in list(self.scalars.keys()):
            match self.normalize_method:
                case 'minmax':
                    # Find the max for each mode across the entire dataset. This is mildly time-consuming,
                    # so we only do it once, then store the scalar and mark the set as normalized_to_max. In
                    # order to make distributions consistent, this step will be required for dist transforms,
                    # so it will be checked before performing the transform

                    # Temporarily turn psuedo_rgb off (if on) so we can save 2/3 memory and not have to worry about dim
                    # shifts for both cases. Temporarily turn off patching so we can save all the additional ops and
                    # just load straight from once. Temporarily turn off normalization so we can load the dataset
                    # without  scaling by scalars as yet unset.
                    temp_psuedo = self.psuedo_rgb
                    self.psuedo_rgb = False
                    temp_patch = self.use_patches
                    self.use_patches = False
                    temp_filter = self.filter_bad_data
                    self.filter_bad_data = False
                    self._normalize_method = None

                    # Preallocate an array. Each row is an individual image, each column is mode
                    maxes = torch.zeros(len(self), self.stack_height, dtype=torch.float32, device=self.device)
                    mins = torch.zeros(len(self), self.stack_height, dtype=torch.float32, device=self.device)
                    for ii, (stack, _) in enumerate(self):
                        # Does the same as np.nanmax(stack, dim=(1,2)) but keeps the tensor on the GPU
                        maxes[ii] = torch.max(
                            torch.max(torch.nan_to_num(stack, nan=-100000), 1).values, 1).values
                        mins[ii] = torch.min(
                            torch.min(torch.nan_to_num(stack, nan=100000), 1).values, 1).values
                    self.scalars['minmax'] = torch.stack(
                        (torch.min(mins, 0).values, torch.max(maxes, 0).values), dim=-1)

                    # Reset psuedo_rgb, patching, and filtering
                    self.psuedo_rgb = temp_psuedo
                    self.use_patches = temp_patch
                    self.filter_bad_data = temp_filter
                    self._normalize_method = 'minmax'

                case 'preset':
                    # Set max value presets for normalization and/or saturation
                    '''
                    These preset values are based on the mean and standard deviation of the dataset for each mode. As a 
                    general rule, when data is normally distributed, 99.7% of samples fall within 3 standard deviations 
                    of the mean. We will use that then as the cutoffs for our presets. Theses were performed for masked
                    data across all modes using the following code:

                        from my_modules.nsclc import NSCLCDataset
                        import numpy as np
                        import torch
                        data = NSCLCDataset(...)
                        x = torch.tensor([])
                        for i in range(len(data)):
                            x = torch.cat((x, data[i][0].unsqueeze(0)), dim=0)
                        x_bar = torch.nanmean(torch.nanmean(torch.nanmean(x, dim=0), dim=1), dim=1)
                        sigma = np.nanstd(np.nanstd(np.nanstd(x, axis=0), axis=1), axis=1)
                        for mode, bar, sig in zip(data.mode, x_bar, sigma):
                            print(f"'{mode}': [{bar}, {sig}],")

                    where 'data' is an instance of this class with all modes (note, not 'all' set as the mode, because 
                    this doesn't actually give all modes)

                    The results for each mode are as  included as a dict below, which will be used to calculate the 
                    ranges for all modes.
                    '''
                    mean_std_mode_dict = {'fad': [0.21998976171016693, 0.0163725633174181],
                                          'nadh': [0.4831325113773346, 0.06519994884729385],
                                          'shg': [0.1766463816165924, 0.15144270658493042],
                                          'intensity': [0.35156112909317017, 0.03592873364686966],
                                          'orr': [0.3561578094959259, 0.022590430453419685],
                                          'g': [0.3222281336784363, 0.009898046031594276],
                                          's': [0.3749162554740906, 0.0024291533045470715],
                                          'photons': [191.2639617919922, 11.133986473083496],
                                          'tau1': [786.095458984375, 51.755043029785156],
                                          'tau2': [4271.404296875, 232.18263244628906],
                                          'alpha1': [103.16101837158203, 6.902841567993164],
                                          'alpha2': [78.77812194824219, 4.96993350982666],
                                          'taumean': [2319.463134765625, 95.37203979492188],
                                          'boundfraction': [0.44960716366767883, 0.01542571373283863]}

                    self._preset_values = {}
                    for key, (m, s) in mean_std_mode_dict.items():
                        self._preset_values[key] = [m - (3 * s), m + (3 * s)]

                    self.scalars['preset'] = torch.tensor(
                        [self._preset_values[mode] for mode in self.mode],
                        dtype=torch.float32, device=self.device)

    # endregion
    # endregion

    # region Show data samples
    def show(self, index, stack_as_rgb=False, cmap='gray'):
        slide_idx, *_ = self.__parse_index__(index)
        if self.dist_transformed:
            fig = plt.figure(index)
            ax = plt.axes()
            ax.plot(self[index][0].T, label=self.mode)
            ax.legend()
            ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
            ax.set_title(f'{self.features['ID']['Slide Name'].iloc[slide_idx]}, {self.label}: {self[index][1]}',
                         fontsize=10)
        else:
            transform = t.ToPILImage()
            if stack_as_rgb:
                fig = plt.figure(index)
                ax = plt.axes()
            else:
                fig, ax = plt.subplots(1, self.stack_height)
            if self.stack_height == 1 or stack_as_rgb:
                img = self[index][0]
                img[torch.isnan(img)] = 0
                ax.imshow(transform(img), cmap=cmap)
                ax.tick_params(top=False, bottom=False, left=False, right=False,
                               labelleft=False, labelbottom=False)
                ax.set_title(f'{self.mode[:]}', fontsize=10)
            else:
                for ii in range(self.stack_height):
                    img = self[index][0][ii]
                    img[torch.isnan(img)] = 0
                    ax[ii].imshow(transform(img), cmap=cmap)
                    ax[ii].tick_params(top=False, bottom=False, left=False, right=False,
                                       labelleft=False, labelbottom=False)
                    ax[ii].set_title(f'{self.mode[ii]}', fontsize=10)
            fig.suptitle(f'{self.features['ID']['Slide Name'].iloc[slide_idx]}, {self.label}: {self[index][1]}',
                         fontsize=10)
            plt.show()
        return fig, ax

    def show_random(self, stack_as_rgb=False, n=5, cmap='gray'):
        for ii in range(n):
            index = np.random.randint(0, len(self))
            self.show(index, stack_as_rgb=stack_as_rgb, cmap=cmap)
    # endregion
