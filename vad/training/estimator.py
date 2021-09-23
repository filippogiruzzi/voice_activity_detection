"""Estimator class for Deep Learning VAD models."""
import tensorflow as tf

from vad.training.model import Resnet1D


class VadEstimator(object):
    """VAD Estimator class.

    Args:
        object ([type]): [description]
    """

    def __init__(self, params):
        """Initialize VadEstimator class parameters.

        Args:
            params (dict): dictionary of model parameters
        """
        self._instantiate_model(params)

    def _instantiate_model(self, params, training=False):
        """Utilitary function to initalize model parameters.

        Args:
            params (dict): dictionary of model parameters
            training (bool, optional): training or inference. Defaults to False.
        """
        if params["model"] == "resnet1d":
            self.model = Resnet1D(params=params, is_training=training)

    def _output_network(self, features, params, training=False):
        """Run model prediction.

        Args:
            features (tf.Tensor): input MFCC features
            params (dict): dictionary of model parameters
            training (bool, optional): training or inference. Defaults to False.

        Returns:
            output (tf.Tensor): output predictions
        """
        self._instantiate_model(params=params, training=training)
        output = self.model(inputs=features)
        return output

    @staticmethod
    def loss_fn(labels, predictions, params):
        """Loss function for training.

        Args:
            labels (dict): dictionary of labels
            predictions (dict): dictionary of predictions
            params (dict): dictionary of model parameters

        Returns:
            loss (tf.Tensor): loss value
        """
        pred = predictions["speech"]
        label = labels["label"]

        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=pred)
        loss = tf.reduce_sum(losses) / params["batch_size"]
        tf.summary.scalar("loss", tensor=loss)
        return loss

    def model_fn(self, features, labels, mode, params):
        """Model wrapper function.

        Args:
            features (tf.Tensor): input MFCC features
            labels (dict): dictionary of labels
            mode (str): model mode, one of ["train", "eval", "infer"]
            params (dict): dictionary of model parameters

        Returns:
            (tf.estimator.EstimatorSpec): ops returned from the model function
        """
        training = mode == tf.estimator.ModeKeys.TRAIN
        preds = self._output_network(features, params, training=training)

        # Training op
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=params["lr"])

            predictions = {"speech": preds}
            loss = self.loss_fn(labels, predictions, params)
            train_op = tf.contrib.training.create_train_op(
                loss, optimizer, global_step=tf.train.get_global_step()
            )

            pred_val = tf.round(tf.nn.sigmoid(predictions["speech"]))
            acc = tf.metrics.accuracy(labels=labels["label"], predictions=pred_val)
            tf.summary.scalar("acc", tensor=acc[1], family="accuracy")
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Evaluation op
        if mode == tf.estimator.ModeKeys.EVAL:
            predictions = {"speech": preds}
            pred_val = tf.round(tf.nn.sigmoid(predictions["speech"]))
            metrics = {
                "accuracy/accuracy/acc": tf.metrics.accuracy(
                    labels=labels["label"],
                    predictions=pred_val,
                )
            }

            loss = self.loss_fn(labels, predictions, params)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=metrics,
            )

        # Prediction op
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                # 'signal_input': features['signal_input'],     # uncomment this line to train.py with --mode predict
                "features_input": features["features_input"],
                "speech": tf.nn.sigmoid(preds),
            }
            export_outputs = {
                "predictions": tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs,
            )
