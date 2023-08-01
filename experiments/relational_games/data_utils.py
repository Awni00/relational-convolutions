import numpy as np
import tensorflow as tf

def get_obj_seq(imgs, obj_indices=tuple(range(9))):
    imgs = imgs.transpose(0,3,1,2)
    imgs = imgs.reshape(len(imgs), 3, 3, 12, 3, 12).transpose(0, 2, 4, 1, 3, 5)
    imgs = imgs.reshape(len(imgs), 9, 3, 12, 12)
    imgs = imgs.transpose(0, 1, 3, 4, 2)
    x = imgs / 255.
    x = x[:, obj_indices]
    return x

# object_slices = []

# for i in range(3):
#     for j in range(3):
#         slice_ = np.s_[:, i*12:(i+1)*12, j*12:(j+1)*12, :]
#         # print(i, j, slice_)
#         object_slices.append(slice_)

# objects = np.stack([imgs[object_slice] for object_slice in object_slices], axis=1)
# objects.shape

def load_ds(filename):
    with np.load(filename) as data:
        imgs = data['images']
        labels = data['labels']

    obj_seqs = get_obj_seq(imgs)
    labels = np.squeeze(labels)

    ds = tf.data.Dataset.from_tensor_slices((obj_seqs, tf.one_hot(labels, 2)))
    del obj_seqs, labels, data

    return ds

def load_task_datasets(task, data_dir):
    if task == 'between':
        file_prefix = '1task_between'
    elif task == 'match_patt':
        file_prefix = '1task_match_patt'
    else:
        raise ValueError(f'invalid task {task}')
    
    task_datasets = {split: load_ds(f'{data_dir}/{file_prefix}_{split}.npz') for split in ('stripes', 'pentos', 'hexos')}
    return task_datasets