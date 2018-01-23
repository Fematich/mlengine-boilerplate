#!/usr/bin/python
import os
import io
import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
from tensorflow_transform.tf_metadata import dataset_schema
from trainer.config import TFRECORD_DIR, FEAT_LEN, BATCH_SIZE, WIDTH, HEIGHT, NUM_LABELS

from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from PIL import Image


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


def read_image(uri):
    """Transforms image into pixel values

    Args:
        uri (str): GCS path were image is stored

    Returns:
        pixels: list of pixel values

    """
    # TF will enable 'rb' in future versions, but until then, 'r' is
    # required.
    def _open_file_read_binary(uri):
      try:
        return file_io.FileIO(uri, mode='rb')
      except errors.InvalidArgumentError:
        return file_io.FileIO(uri, mode='r')

    try:
      with _open_file_read_binary(uri) as f:
        image_bytes = f.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((WIDTH,HEIGHT))

    # A variety of different calling libraries throw different exceptions here.
    # They all correspond to an unreadable file so we treat them equivalently.
    except Exception as e:  # pylint: disable=broad-except
      logging.exception('Error processing image %s: %s', uri, str(e))
      return

    img_data = img.getdata()
    pixels = np.asarray(img_data, dtype=float) / 255

    return pixels.flatten().tolist()


schema = dataset_schema.Schema(
    {
        'id': dataset_schema.ColumnSchema(tf.string, [],
                                          dataset_schema.FixedColumnRepresentation()),
        'label': dataset_schema.ColumnSchema(tf.int64, [NUM_LABELS],
                                             dataset_schema.FixedColumnRepresentation()),
        'feat': dataset_schema.ColumnSchema(tf.float32, [FEAT_LEN],
                                            dataset_schema.FixedColumnRepresentation())
    }
)