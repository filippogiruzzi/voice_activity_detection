import os
from absl import flags, app
import time

import numpy as np
import tensorflow as tf
from collections import deque
import seaborn as sns
import matplotlib.pyplot as plt

from vad.data_processing.data_iterator import split_data, file_iter
from vad.data_processing.feature_extraction import extract_features
from vad.training.input_pipeline import FEAT_SIZE


flags.DEFINE_string('data_dir',
                    '/home/filippo/datasets/LibriSpeech/',
                    'path to data directory')
flags.DEFINE_string('exported_model',
                    '/home/filippo/datasets/LibriSpeech/tfrecords/models/resnet1d/inference/exported/',
                    'path to pretrained TensorFlow exported model')
flags.DEFINE_integer('seq_len',
                     1024,
                     'sequence length for speech prediction')
flags.DEFINE_integer('stride',
                     1,
                     'stride for sliding window prediction')
flags.DEFINE_boolean('smoothing',
                     False,
                     'apply smoothing feature')
FLAGS = flags.FLAGS


def visualize_predictions(signal, fn, preds, sr=16000):
    fig = plt.figure(figsize=(15, 10))
    sns.set()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([i / sr for i in range(len(signal))], signal)
    for predictions in preds:
        color = 'r' if predictions[2] == 0 else 'g'
        ax.axvspan((predictions[0]) / sr, predictions[1] / sr, alpha=0.5, color=color)
    plt.title('Prediction on signal {}, speech in green'.format(fn), size=20)
    plt.xlabel('Time (s)', size=20)
    plt.ylabel('Amplitude', size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()


def smooth_predictions(preds):
    smoothed_preds = []
    # Smooth with 3 consecutive windows
    for i in range(2, len(preds), 3):
        cur_pred = preds[i]
        if cur_pred[2] == preds[i - 1][2] == preds[i - 2][2]:
            smoothed_preds.append([preds[i - 2][0], cur_pred[1], cur_pred[2]])
        else:
            if len(smoothed_preds) > 0:
                smoothed_preds.append([preds[i - 2][0], cur_pred[1], smoothed_preds[-1][2]])
            else:
                smoothed_preds.append([preds[i - 2][0], cur_pred[1], 0.0])
    # Hangover
    n = 0
    while n < len(smoothed_preds):
        cur_pred = smoothed_preds[n]
        if cur_pred[2] == 1:
            if n > 0:
                smoothed_preds[n - 1][2] = 1
            if n < len(smoothed_preds) - 1:
                smoothed_preds[n + 1][2] = 1
            n += 2
        else:
            n += 1
    return smoothed_preds


def main(_):
    np.random.seed(0)

    # Directories
    data_dir = os.path.join(FLAGS.data_dir, 'test-clean/')
    label_dir = os.path.join(FLAGS.data_dir, 'labels/')

    _, _, test = split_data(label_dir, split='0.7/0.15', random_seed=0)
    file_it = file_iter(data_dir, label_dir, files=test)

    # TensorFlow inputs
    features_input_ph = tf.placeholder(shape=FEAT_SIZE, dtype=tf.float32)
    features_input_op = tf.transpose(features_input_ph, perm=[1, 0])
    features_input_op = tf.expand_dims(features_input_op, axis=0)

    # TensorFlow exported model
    speech_predictor = tf.contrib.predictor.from_saved_model(export_dir=FLAGS.exported_model)
    init = tf.initializers.global_variables()
    classes = ['Noise', 'Speech']

    # Iterate though test data
    with tf.Session() as sess:
        for signal, labels, fn in file_it:
            sess.run(init)
            print('\nPrediction on file {} ...'.format(fn))
            signal_input = deque(signal[:FLAGS.seq_len].tolist(), maxlen=FLAGS.seq_len)

            preds, pred_time = [], []
            pointer = FLAGS.seq_len
            while pointer < len(signal):
                start = time.time()
                # Preprocess signal & extract features
                signal_to_process = np.copy(signal_input)
                signal_to_process = np.float32(signal_to_process)
                features = extract_features(signal_to_process, freq=16000, n_mfcc=5, size=512, step=16)

                # Prediction
                features_input = sess.run(features_input_op, feed_dict={features_input_ph: features})
                speech_prob = speech_predictor({'features_input': features_input})['speech'][0]
                speech_pred = classes[int(np.round(speech_prob))]

                # Time prediction & processing
                end = time.time()
                dt = end - start
                pred_time.append(dt)
                print('Prediction = {} | proba = {:.2f} | time = {:.2f} s'.format(speech_pred, speech_prob[0], dt))

                # For visualization
                preds.append([pointer - FLAGS.seq_len, pointer, np.round(speech_prob)])

                # Update signal segment
                signal_input.extend(signal[pointer + FLAGS.stride:pointer + FLAGS.stride + FLAGS.seq_len])
                pointer += FLAGS.seq_len + FLAGS.stride

            print('Average prediction time = {:.2f} ms'.format(np.mean(pred_time) * 1e3))

            # Smoothing & hangover
            if FLAGS.smoothing:
                preds = smooth_predictions(preds)

            # Visualization
            visualize_predictions(signal, fn, preds)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
