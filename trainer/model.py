#!/usr/bin/python
import tensorflow as tf


def inference(features):
    """Creates the predictions of the model

        Args:
          features (dict): A dictionary of tensors keyed by the feature name.

        Returns:
            A tensor that represents the predictions

    """
    with tf.variable_scope('denselayer'):
        print(features['feat'].get_shape())
        predictions = tf.layers.dense(inputs=features['feat'],
                                      units=1,
                                      name='dense_weights',
                                      use_bias=True)
        predictions_squeezed = tf.squeeze(predictions)
    return predictions_squeezed


def loss(predictions, labels):
    """Function that calculates the loss based on the predictions and labels

        Args:
          predictions: A tensor representing the predictions (output from)
          labels: A tensor representing the labels.

        Returns:
            A tensor representing the loss

    """
    with tf.variable_scope('loss'):
        return tf.losses.mean_squared_error(predictions, labels)


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
                global_step=tf.train.framework.get_global_step(),
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