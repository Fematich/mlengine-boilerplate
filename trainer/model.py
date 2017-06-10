#!/usr/bin/python
import tensorflow as tf


def inference(features):
    '''
    Creates the predictions of the model

        Args:
          features: A dictionary of tensors keyed by the feature name.

        Returns:
          A tensor that represents the predictions
    '''
    with tf.variable_scope('denselayer') as scope:
        print(features['feat'].get_shape())
        predictions = tf.layers.dense(inputs=features['feat'],
                                      units=1, name='dense_weights', use_bias=True)
        predictions_squeezed = tf.squeeze(predictions)
    return predictions_squeezed


def loss(predictions, labels):
    '''
    Function that calculates the loss based on the predictions and labels

        Args:
          predictions: A tensor representing the predictions (output from)
          labels: A tensor representing the labels.

        Returns:
          A tensor representing the loss
    '''
    with tf.variable_scope('loss') as scope:
        loss = tf.losses.mean_squared_error(
            predictions, labels)
    return loss


def build_model_fn():
    def _model_fn(features, labels, mode, params):
        '''
        Creates the prediction and its loss.

        Args:
          features: A dictionary of tensors keyed by the feature name.
          labels: A tensor representing the labels.
          mode: The execution mode, defined in tf.contrib.learn.ModeKeys.

        Returns:
          A tuple consisting of the prediction, loss, and train_op.
        '''
        predictions = inference(features)
        if mode == tf.contrib.learn.ModeKeys.INFER:
            return predictions, None, None

        loss_op = loss(predictions, labels)
        if mode == tf.contrib.learn.ModeKeys.EVAL:
            return predictions, loss_op, None

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss_op,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params['learning_rate'],
            optimizer='Adagrad',
            summaries=[
                'learning_rate',
                'loss',
                'gradients',
                'gradient_norm',
            ],
            name='train')

        return predictions, loss_op, train_op

    return _model_fn
