#!/usr/bin/python
import logging

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils

from model import build_model_fn
from util import read_data
from config import MODEL_DIR, FEAT_LEN

if __name__ == '__main__':
    estimator = tf.contrib.learn.Estimator(
        model_fn=build_model_fn(),
        model_dir=MODEL_DIR,
        params={'learning_rate': 0.001},)
    estimator.fit(input_fn=read_data, max_steps=100)

    def serving_input_fn():
        """Builds the input subgraph for prediction.
        Source: http://bit.ly/2rgXgBK
        This serving_input_fn accepts raw Tensors inputs which will be fed to the
        server as JSON dictionaries. The values in the JSON dictionary will be
        converted to Tensors of the appropriate type.
        Returns:
           tf.contrib.learn.input_fn_utils.InputFnOps, a named tuple
           (features, labels, inputs) where features is a dict of features to be
           passed to the Estimator, labels is always None for prediction, and
           inputs is a dictionary of inputs that the prediction server should expect
           from the user.
        """

        feature_placeholders = {
            'id': tf.placeholder(tf.string, [None], name='id_placeholder'),
            'feat': tf.placeholder(tf.float32, [None, FEAT_LEN], name='feat_placeholder'),
            # label is not required since serving is only used for inference
        }
        return input_fn_utils.InputFnOps(
            feature_placeholders,
            None,
            feature_placeholders
        )
    _ = estimator.export_savedmodel(
        MODEL_DIR, serving_input_fn)
