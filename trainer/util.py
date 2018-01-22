#!/usr/bin/python
import os

import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
from tensorflow_transform.tf_metadata import dataset_schema
from trainer.config import TFRECORD_DIR, FEAT_LEN, BATCH_SIZE


def build_input_fn(mode='train'):
    """Builds input function for estimator.

    Args:
        mode (str): mode estimator is running in

    Returns:
        function: input function

    """
    def _input_fn(data_dir=TFRECORD_DIR, batch_size=BATCH_SIZE):
        """Input function for estimator

        Args:
            data_dir (str): GCS directory where tfrecords are stored
            batch_size (int): batch size to use for reading

        Returns:
            features, label

        """
        def gzip_reader():
            return tf.TFRecordReader(
                options=tf.python_io.TFRecordOptions(
                    compression_type=TFRecordCompressionType.GZIP))

        features = tf.contrib.learn.io.read_batch_features(
            file_pattern=os.path.join(data_dir, mode + '*'),
            batch_size=batch_size,
            reader=gzip_reader,
            features={
                'id': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.float32),
                'feat': tf.FixedLenFeature([FEAT_LEN], tf.float32),
            })

        label = features.pop('label')
        return features, label

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