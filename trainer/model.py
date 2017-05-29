#!/usr/bin/python
def build_model_fn(self):
    def _model_fn(features, labels, mode, params):
        """Creates the prediction and its loss.

        Args:
          features: A dictionary of tensors keyed by the feature name.
          labels: A tensor representing the labels.
          mode: The execution mode, defined in tf.contrib.learn.ModeKeys.

        Returns:
          A tuple consisting of the prediction, loss, and train_op.
        """
        predictions = self.inference(features)
        if mode == tf.contrib.learn.ModeKeys.INFER:
            return predictions, None, None

        loss = self.loss(predictions, labels)
        if mode == tf.contrib.learn.ModeKeys.EVAL:
            return predictions, loss, None

        non_static_variables = set(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        for name, concept in self.concepts.items():
            if not concept.target:
                non_static_variables = non_static_variables - set(
                    tf.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES,
                        scope=concept.__class__.__name__.lower()))
        non_static_variables = list(non_static_variables)


        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params["learning_rate"],
            optimizer='Adagrad',
            summaries=[
                "learning_rate",
                "loss",
                "gradients",
                "gradient_norm",
            ],
            name='train')

        if params["static_concepts"]:
            return predictions, loss, train_op_static
        else:
            return predictions, loss, train_op

    return _model_fn
