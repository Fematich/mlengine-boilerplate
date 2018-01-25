#!/usr/bin/python
import tensorflow as tf

import util
from model import build_model_fn
from config import MODEL_DIR, FEAT_LEN

if __name__ == '__main__':
    estimator = tf.estimator.Estimator(
        model_fn=build_model_fn(),
        model_dir=MODEL_DIR,
        params={'learning_rate': 0.001})

    train_spec = tf.estimator.TrainSpec(input_fn=util.build_input_fn(), max_steps=5000)
    eval_spec = tf.estimator.EvalSpec(input_fn=util.build_input_fn(mode='test'), steps=300)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    feature_placeholders = {
        'id': tf.placeholder(tf.string, [None], name='id_placeholder'),
        'feat': tf.placeholder(tf.float32, [None, FEAT_LEN],
                               name='feat_placeholder'),
        # label is not required since serving is only used for inference
    }
    
    serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        feature_placeholders)

    estimator.export_savedmodel(
        MODEL_DIR, serving_input_fn)