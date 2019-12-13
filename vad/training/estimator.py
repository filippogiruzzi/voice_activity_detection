import tensorflow as tf

from vad.training.model import Resnet1D


class VadEstimator(object):
    def __init__(self, params):
        self._instantiate_model(params)

    def _instantiate_model(self, params, training=False):
        if params['model'] == 'resnet1d':
            self.model = Resnet1D(params=params, is_training=training)

    def _output_network(self, features, params, training=False):
        self._instantiate_model(params=params, training=training)
        output = self.model(inputs=features)
        return output

    @staticmethod
    def loss_fn(labels, predictions, params):
        pred = predictions['speech']
        label = labels['label']

        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=pred)
        loss = tf.reduce_sum(losses) / params['batch_size']
        tf.summary.scalar('loss', tensor=loss)
        return loss

    def model_fn(self, features, labels, mode, params):
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        preds = self._output_network(features, params, training=training)

        # Training op
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])

            predictions = {'speech': preds}
            loss = self.loss_fn(labels, predictions, params)
            train_op = tf.contrib.training.create_train_op(loss, optimizer, global_step=tf.train.get_global_step())

            pred_val = tf.round(tf.nn.sigmoid(predictions['speech']))
            acc = tf.metrics.accuracy(labels=labels['label'], predictions=pred_val)
            tf.summary.scalar('acc', tensor=acc[1], family='accuracy')
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Evaluation op
        if mode == tf.estimator.ModeKeys.EVAL:
            predictions = {'speech': preds}
            pred_val = tf.round(tf.nn.sigmoid(predictions['speech']))
            metrics = {
                'accuracy/accuracy/acc': tf.metrics.accuracy(labels=labels['label'], predictions=pred_val)
            }

            loss = self.loss_fn(labels, predictions, params)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)

        # Prediction op
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                # 'signal_input': features['signal_input'],     # uncomment this line to train.py with --mode predict
                'features_input': features['features_input'],
                'speech': tf.nn.sigmoid(preds)
            }
            export_outputs = {'predictions': tf.estimator.export.PredictOutput(predictions)}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
