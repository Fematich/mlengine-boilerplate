#!/usr/bin/python
import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_schema
from .config import DATA_DIR, FEAT_LEN, BATCH_SIZE


def read_data(datadir=DATA_DIR, batch_size=BATCH_SIZE, mode='train'):
    """
    Creates an input_function for the astrohack data
    """
    def gzip_reader():
        return tf.TFRecordReader(
            options=tf.python_io.TFRecordOptions(
                compression_type=TFRecordCompressionType.GZIP))
    features = tf.contrib.learn.io.read_batch_features(
        file_pattern=os.path.join(datadir, mode),
        batch_size=batch_size,
        reader=gzip_reader,
        features={
            'id': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.float32),
            'feat': tf.FixedLenFeature([FEAT_LEN], tf.float32),
        })

    label = features.pop('label')
    return features, label

schema = dataset_schema.Schema({
    'id': dataset_schema.ColumnSchema(tf.string, [],
                                      dataset_schema.FixedColumnRepresentation()),
    'label': dataset_schema.ColumnSchema(tf.float32, [],
                                         dataset_schema.FixedColumnRepresentation()),
    'feat': dataset_schema.ColumnSchema(tf.float32, [FEAT_LEN],
                                        dataset_schema.FixedColumnRepresentation())
})
