#!/usr/bin/python
import logging

import tensorflow as tf

from model import build_model_fn
from util import read_data
from config import MODEL_DIR, FEAT_LEN

if __name__ == '__main__':
    estimator = tf.estimator.Estimator(
        model_fn=build_model_fn(),
        model_dir=MODEL_DIR,
        params={'learning_rate': 0.001},)
    estimator.train(input_fn=read_data, max_steps=100)

    result = estimator.evaluate(input_fn=lambda: read_data(mode='test'), steps=300)
    #print result

    feature_placeholders = {
        'id': tf.placeholder(tf.string, [None], name='id_placeholder'),
        'feat': tf.placeholder(tf.float32, [None, FEAT_LEN], name='feat_placeholder'),
        # label is not required since serving is only used for inference
    }
    
    serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        feature_placeholders)

    estimator.export_savedmodel(
        MODEL_DIR, serving_input_fn)
