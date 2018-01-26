#!/usr/bin/python
import tensorflow as tf
from config import NUM_LABELS


def inference(features):
    """Creates the predictions of the model

        Args:
          features (dict): A dictionary of tensors keyed by the feature name.

        Returns:
            A tensor that represents the predictions

    """
    # five layers and their number of neurons (the last layer has NUM_LABELS softmax neurons)
    L = 1000
    M = 300
    N = 150
    O = 30

    with tf.variable_scope('denselayer'):
        # print(type(features['feat']))
        # print(features['feat'].get_shape())
        layer1 = tf.layers.dense(inputs=features['feat'],
                                      units=L,
                                      name='layer1',
                                      use_bias=True)
        relu_layer1 = tf.nn.relu(layer1)
        layer2 = tf.layers.dense(inputs=relu_layer1,
                                      units=M,
                                      name='layer2',
                                      use_bias=True)
        relu_layer2= tf.nn.relu(layer2)
        layer3 = tf.layers.dense(inputs=relu_layer2,
                                      units=N,
                                      name='layer3',
                                      use_bias=True)
        relu_layer3 = tf.nn.relu(layer3)
        layer4 = tf.layers.dense(inputs=relu_layer3,
                                      units=O,
                                      name='layer4',
                                      use_bias=True)
        relu_layer4 = tf.nn.relu(layer4)
        layer5 = tf.layers.dense(inputs=relu_layer4,
                                      units=NUM_LABELS,
                                      name='layer5',
                                      use_bias=True)
        predictions = tf.nn.softmax(layer5)

    return predictions


def loss(predictions, labels):
    """Function that calculates the loss based on the predictions and labels

        Args:
          predictions: A tensor representing the predictions (output from)
          labels: A tensor representing the labels.

        Returns:
            A tensor representing the loss

    """
    with tf.variable_scope('loss'):
        return tf.losses.mean_squared_error(labels, predictions)


def build_model_fn():
    """Build model function as input for estimator.

    Returns:
        function: model function

    """
    def _model_fn(features, labels, mode, params):
        """Creates the prediction and its loss.

        Args:
          features (dict): A dictionary of tensors keyed by the feature name.
          labels: A tensor representing the labels.
          mode: The execution mode, defined in tf.estimator.ModeKeys.

        Returns:
          tf.estimator.EstimatorSpec: EstimatorSpec object containing mode,
          predictions, loss, train_op and export_outputs.

        """
        predictions = inference(features)
        loss_op = None
        train_op = None
        
        if mode != tf.estimator.ModeKeys.PREDICT:
            loss_op = loss(predictions, labels)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss_op,
                global_step=tf.train.get_global_step(),
                learning_rate=params['learning_rate'],
                optimizer='Adagrad',
                summaries=[
                    'learning_rate',
                    'loss',
                    'gradients',
                    'gradient_norm',
                ],
                name='train')

        predictions_dict = {"predictions": predictions}
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: 
                tf.estimator.export.PredictOutput(predictions_dict)}


        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions_dict,
            loss=loss_op, 
            train_op=train_op,
            export_outputs=export_outputs)

    return _model_fn
