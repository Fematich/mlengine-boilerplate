#!/usr/bin/python
import logging

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
from tensorflow.contrib.layers import create_feature_spec_for_parsing

from .model import model_fn
from .util import schema


if __name__ == "__main__":
    estimator = tf.contrib.learn.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.output_dir,
        params={"learning_rate": 0.001},)
    estimator.fit(input_fn=read_astro, max_steps=100000)

    # Export model
    # (https://github.com/MtDersvan/tf_playground/blob/master/wide_and_deep_tutorial/wide_and_deep_basic_serving.md)
    schema.pop('label')
    serving_input_fn = input_fn_utils.build_parsing_serving_input_fn(
        schema)  #TODO: validate this input, otherwise use format from http://bit.ly/2rgXgBK
    _ = estimator.export_savedmodel(
        FLAGS.output_dir, serving_input_fn)
