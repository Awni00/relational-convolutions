import numpy as np
import tensorflow as tf
from tqdm import tqdm

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

def load_ds_from_npz(filename):
    with np.load(filename) as data:
        imgs = data['images']
        labels = data['labels']

    obj_seqs = get_obj_seq(imgs)
    labels = np.squeeze(labels)

    ds = tf.data.Dataset.from_tensor_slices((obj_seqs, tf.one_hot(labels, 2)))
    del obj_seqs, labels, data

    return ds

def load_ds_from_tfrecord(tfrecord_filename):
    def parse_example(example_proto):
        feature_description = {
            'x': tf.io.FixedLenFeature([], tf.string),
            'y': tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(example_proto, feature_description)
        x = tf.io.parse_tensor(example['x'], out_type=tf.float64)
        y = tf.io.parse_tensor(example['y'], out_type=tf.float32)
        return x, y

    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    dataset = dataset.map(parse_example)
    return dataset

def get_shape_from_ds(ds):
    x, y = dataset.take(1).__iter__().next()
    return x.shape, y.shape

def write_ds_to_tfrecord(dataset, rfrecord_filename):
    with tf.io.TFRecordWriter(rfrecord_filename) as writer:
        for x, y in tqdm(dataset):
            # Serialize the input and output tensors
            x_serialized = tf.io.serialize_tensor(x)
            y_serialized = tf.io.serialize_tensor(y)

            # dictionary representing the example
            example = {
                'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_serialized.numpy()])),
                'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_serialized.numpy()]))
            }

            # create features message using example
            features = tf.train.Features(feature=example)
            example_proto = tf.train.Example(features=features)

            # serialize to string and write example
            writer.write(example_proto.SerializeToString())

def load_task_datasets(task, data_dir, data_format='tfrecord'):
    if task == 'between':
        file_prefix = '1task_between'
    elif task == 'match_patt':
        file_prefix = '1task_match_patt'
    else:
        raise ValueError(f'invalid task {task}')
    
    if data_format=='npz':
        task_datasets = {split: load_ds_from_npz(f'{data_dir}/{file_prefix}_{split}.npz') for split in ('stripes', 'pentos', 'hexos')}
    elif data_format == 'tfrecord':
        task_datasets = {split: load_ds_from_tfrecord(f'{data_dir}/{file_prefix}_{split}.tfrecord') for split in ('stripes', 'pentos', 'hexos')}
    else:
        raise ValueError(f'invalid data_format {data_format}')

    return task_datasets