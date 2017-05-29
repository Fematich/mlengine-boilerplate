#!/usr/bin/python
import tensorflow as tf
from .model import model_fn


if __name__ == "__main__":
    estimator = tf.contrib.learn.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.output_dir)
    estimator.fit(input_fn=read_astro, max_steps=100000)
#TODO save model