#!/usr/bin/python
import multiprocessing

import tensorflow as tf
from tensorflow import data
from tensorflow_transform.tf_metadata import dataset_schema

from trainer.config import TFRECORD_DIR, FEAT_LEN, BATCH_SIZE

def get_features_target_tuple(features):
    """
        Get a tuple of input feature tensors and target feature tensor.

        Args:
            features
        Returns:
             tuple of feature and target feature tensors
    """
    label = features.pop('label')
    return features, label

def build_input_fn(mode=tf.estimator.ModeKeys.TRAIN, multi_threading=True, num_epochs=20):
    """Builds input function for estimator.

    Args:
        mode (str): mode estimator is running in
        multi_threading (bool):  indicating if multi threading should be enabled
        num_epochs (int): number of epochs to go over the dataset
    Returns:
        function: input function

    """
    def _input_fn(data_dir=TFRECORD_DIR, batch_size=BATCH_SIZE):
        shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
        num_threads = multiprocessing.cpu_count() if multi_threading else 1
        buffer_size = 2 * batch_size + 1
        file_names = tf.matching_files(data_dir)
        feature_spec = {
                'id': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.float32),
                'feat': tf.FixedLenFeature([FEAT_LEN], tf.float32),
            }
        dataset = data.TFRecordDataset(filenames=file_names, compression_type = 'GZIP')
        dataset = dataset.map(lambda tf_example: printout(tf_example))
        dataset = dataset.map(lambda tf_example:  tf.parse_example(serialized=[tf_example], features=feature_spec),
                              num_parallel_calls=num_threads)

        dataset = dataset.map(lambda features: get_features_target_tuple(features),
                              num_parallel_calls=num_threads)

        if shuffle:
            dataset = dataset.shuffle(buffer_size)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size)
        dataset = dataset.repeat(num_epochs)

        iterator = dataset.make_one_shot_iterator()
        features, target = iterator.get_next()

        return features, target

    return _input_fn


schema = dataset_schema.Schema(
    {
        'id': dataset_schema.ColumnSchema(tf.string, [],
                                          dataset_schema.FixedColumnRepresentation()),
        'label': dataset_schema.ColumnSchema(tf.float32, [],
                                             dataset_schema.FixedColumnRepresentation()),
        'feat': dataset_schema.ColumnSchema(tf.float32, [FEAT_LEN],
                                            dataset_schema.FixedColumnRepresentation())
    }
)