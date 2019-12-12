import os
import multiprocessing
from absl import flags, app
from copy import deepcopy

from tqdm import tqdm
import tensorflow as tf
import numpy as np

from vad.data_processing.data_iterator import split_data, slice_iter
from vad.data_processing.feature_extraction import extract_features


flags.DEFINE_string('data_dir',
                    '/home/filippo/datasets/LibriSpeech/',
                    'data directory path')
flags.DEFINE_string('data_split',
                    '0.7/0.15',
                    'train/val split')
flags.DEFINE_integer('seq_len',
                     1024,
                     'segment length')
flags.DEFINE_integer('num_shards',
                     256,
                     'number of tfrecord files')
flags.DEFINE_boolean('debug',
                     False,
                     'debug for a few samples')
flags.DEFINE_string('data_type',
                    'trainvaltest',
                    'data types to write into tfrecords')

FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(file_id,
                      start,
                      end,
                      sub_segment,
                      sub_segment_id,
                      sub_segment_len,
                      label):
    # Drop 1/4th of speech signals to balance data
    if label == 1:
        drop = np.random.randint(0, 4)
        if drop > 0:
            return None

    # MFCC feature extraction
    signal_to_process = np.copy(sub_segment)
    signal_to_process = np.float32(signal_to_process)
    features = extract_features(signal_to_process, freq=16000, n_mfcc=5, size=512, step=16)
    features = np.reshape(features, -1)

    feature_dict = {
        'signal/id': bytes_feature(file_id.encode()),
        'segment/start': int64_feature(int(start)),
        'segment/end': int64_feature(int(end)),
        'subsegment/id': int64_feature(sub_segment_id),
        'subsegment/length': int64_feature(sub_segment_len),
        'subsegment/signal': float_list_feature(sub_segment.tolist()),
        'subsegment/features': float_list_feature(features.tolist()),
        'subsegment/label': int64_feature(label),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def pool_create_tf_example(args):
    return create_tf_example(*args)


def write_tfrecords(path, dataiter, num_shards=256, nmax=-1):
    writers = [
        tf.python_io.TFRecordWriter('{}{:05d}_{:05d}.tfrecord'.format(path, i, num_shards)) for i in range(num_shards)
    ]
    print('\nWriting to output path: {}'.format(path))
    pool = multiprocessing.Pool()
    counter = 0
    for i, tf_example in tqdm(enumerate(pool.imap(pool_create_tf_example, [(deepcopy(data['file_id']),
                                                                            deepcopy(data['start']),
                                                                            deepcopy(data['end']),
                                                                            deepcopy(data['sub_segment']),
                                                                            deepcopy(data['sub_segment_id']),
                                                                            deepcopy(data['sub_segment_len']),
                                                                            deepcopy(data['label'])
                                                                            ) for data in dataiter]))):
        if tf_example is not None:
            writers[i % num_shards].write(tf_example.SerializeToString())
            counter += 1
        if 0 < nmax < i:
            break
    pool.close()
    for writer in writers:
        writer.close()
    print('Recorded {} signals'.format(counter))


def create_tfrecords(data_dir,
                     seq_len=1024,
                     split='0.7/0.15',
                     num_shards=256,
                     debug=False,
                     data_type='trainval'):
    np.random.seed(0)

    output_path = os.path.join(data_dir, 'tfrecords/')
    if not tf.gfile.IsDirectory(output_path):
        tf.gfile.MakeDirs(output_path)

    # Data & label directories
    label_dir = os.path.join(data_dir, 'labels/')
    data_dir = os.path.join(data_dir, 'test-clean/')

    # Split data on files
    train, val, test = split_data(label_dir, split, random_seed=0)

    print('\nTotal files: {}'.format(len(train) + len(val) + len(test)))
    print('Train/val/test split: {}/{}/{}'.format(len(train), len(val), len(test)))
    train_it = slice_iter(data_dir, label_dir, train, seq_len)
    val_it = slice_iter(data_dir, label_dir, val, seq_len)
    test_it = slice_iter(data_dir, label_dir, test, seq_len)

    # Write data to tfrecords format
    nmax = 100 if debug else -1
    if 'train' in data_type:
        print('\nWriting train tfrecords ...')
        train_path = os.path.join(output_path, 'train/')
        if not tf.gfile.IsDirectory(train_path):
            tf.gfile.MakeDirs(train_path)
        write_tfrecords(train_path, train_it, num_shards, nmax=nmax)

    if 'val' in data_type:
        print('\nWriting val tfrecords ...')
        val_path = os.path.join(output_path, 'val/')
        if not tf.gfile.IsDirectory(val_path):
            tf.gfile.MakeDirs(val_path)
        write_tfrecords(val_path, val_it, num_shards, nmax=nmax)

    if 'test' in data_type:
        print('\nWriting test tfrecords ...')
        test_path = os.path.join(output_path, 'test/')
        if not tf.gfile.IsDirectory(test_path):
            tf.gfile.MakeDirs(test_path)
        write_tfrecords(test_path, test_it, num_shards, nmax=nmax)


def main(_):
    create_tfrecords(FLAGS.data_dir,
                     seq_len=FLAGS.seq_len,
                     split=FLAGS.data_split,
                     num_shards=FLAGS.num_shards,
                     debug=FLAGS.debug,
                     data_type=FLAGS.data_type)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
