import numpy as np
import tensorflow as tf
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import data_utils
from tqdm import tqdm, trange

import gc
gc.enable(); gc.set_threshold(0)

data_path = '../../data/relational_games'

# get the file paths for all .npz files in the directory
# npz_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npz')]

tasks = ('1task_between', '1task_match_patt')
splits = ('hexos', 'pentos', 'stripes')

filename_prefixes = [f'{task}_{split}' for task in tasks for split in splits]

npz_files = [f'{data_path}/{filename_prefix}.npz' for filename_prefix in filename_prefixes]


dataspec_path = f'{data_path}/dataset_specs.npy'
if os.path.exists(dataspec_path):
    dataset_specs = np.load(dataspec_path, allow_pickle=True).item()
else:
    dataset_specs = {}

def get_ds_spec(ds):
    ds_card = len(ds)
    x_shape = ds.element_spec[0].shape
    y_shape = ds.element_spec[1].shape
    return {'ds_card': ds_card, 'x_shape': x_shape, 'y_shape': y_shape}


# Convert each .npz file to a .tfrecord file
for npz_file in tqdm(npz_files):
    filename_prefix = os.path.splitext(os.path.basename(npz_file))[0]
    tfrecord_file = f'{data_path}/{filename_prefix}.tfrecord'

    if not os.path.exists(tfrecord_file):
        npz_ds = data_utils.load_ds_from_npz(npz_file)
        dataset_specs[filename_prefix] = get_ds_spec(npz_ds)
        data_utils.write_ds_to_tfrecord(npz_ds, f'{data_path}/{filename_prefix}.tfrecord')
        del npz_ds

np.save(dataspec_path, dataset_specs, allow_pickle=True)