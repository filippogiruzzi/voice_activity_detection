"""Main entrypoint to train Deep Learning VAD models."""
import argparse
import glob
import logging
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from vad.training.estimator import VadEstimator
from vad.training.input_pipeline import data_input_fn


def train(
    params,
    data_dir,
    model_dir=None,
    ckpt=None,
    mode="train",
    fake_input=False,
):
    """Train a CNN for Voice Activity Detection on a TFRecords dataset.

    Args:
        params (dict): dictionary of model training parameters
        data_dir (str): path to TFRecords dataset directory
        model_dir (str, optional): path to trained model directory. Defaults to None
        ckpt (str, optional): path to pre-trained checkpoint directory. Default to None
        mode (str, optional): TF estimator mode, one of ["train", "eval", "predict"]. Defaults to "train"
        fake_input (bool, optional): debugging option to train on 1 batch. Defaults to False
    """
    tfrecords_train = glob.glob(f"{data_dir}train/*.tfrecord")
    tfrecords_val = glob.glob(f"{data_dir}val/*.tfrecord")
    tfrecords_test = glob.glob(f"{data_dir}test/*.tfrecord")

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    logger = logging.getLogger(__name__)
    np.random.seed(0)
    tf.set_random_seed(0)
    tf.logging.set_verbosity(tf.logging.INFO)

    model = params["model"]
    input_size = params["input_size"]
    subsample = params["subsample"]

    if not model_dir:
        save_dir = f"{data_dir}models/{model}/{datetime.now().isoformat()}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = model_dir

    train_config = tf.estimator.RunConfig(
        save_summary_steps=10,
        save_checkpoints_steps=500,
        keep_checkpoint_max=20,
        log_step_count_steps=10,
    )

    ws = None
    if ckpt:
        ws = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=ckpt,
            vars_to_warm_start=".*",
        )

    # Create TensorFlow estimator object
    estimator_obj = VadEstimator(params)
    estimator = tf.estimator.Estimator(
        model_fn=estimator_obj.model_fn,
        model_dir=save_dir,
        config=train_config,
        params=params,
        warm_start_from=ws,
    )

    mode_keys = {
        "train": tf.estimator.ModeKeys.TRAIN,
        "eval": tf.estimator.ModeKeys.EVAL,
        "predict": tf.estimator.ModeKeys.PREDICT,
    }
    mode = mode_keys[mode]

    # Training & Evaluation on Train / Val set
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_input_fn = data_input_fn(
            tfrecords_train,
            batch_size=params["batch_size"],
            epochs=1,
            input_size=input_size,
            n_classes=params["n_classes"],
            subsample=subsample,
            shuffle=True,
            fake_input=fake_input,
        )
        eval_input_fn = data_input_fn(
            tfrecords_val,
            batch_size=params["batch_size"],
            epochs=1,
            input_size=input_size,
            n_classes=params["n_classes"],
            subsample=subsample,
            shuffle=False,
            fake_input=fake_input,
        )

        for epoch_num in range(params["epochs"]):
            logger.info(f"Training for epoch {epoch_num} ...")
            estimator.train(input_fn=train_input_fn)
            logger.info(f"Evaluation for epoch {epoch_num} ...")
            estimator.evaluate(input_fn=eval_input_fn)

    # Evaluation on Test set
    elif mode == tf.estimator.ModeKeys.EVAL:
        test_input_fn = data_input_fn(
            tfrecords_val,
            batch_size=params["batch_size"],
            epochs=1,
            input_size=input_size,
            n_classes=params["n_classes"],
            subsample=subsample,
            shuffle=False,
            fake_input=fake_input,
        )

        logger.info("Evaluation of test set ...")
        estimator.evaluate(input_fn=test_input_fn)

    # Prediction visualization on Test set
    elif mode == tf.estimator.ModeKeys.PREDICT:
        test_input_fn = data_input_fn(
            tfrecords_test,
            batch_size=params["batch_size"],
            epochs=1,
            input_size=input_size,
            n_classes=params["n_classes"],
            subsample=subsample,
            shuffle=False,
            fake_input=fake_input,
        )

        classes = ["Noise", "Speech"]
        predictions = estimator.predict(input_fn=test_input_fn)
        for n, pred in enumerate(predictions):
            signal_input = pred["signal_input"]
            pred = pred["speech"]

            # Plot signal (uncomment # 'signal_input': features['signal_input'] in estimator.py)
            sns.set()
            sns.lineplot(
                x=[i for i in range(len(signal_input[:, 0]))],
                y=signal_input[:, 0],
            )
            plt.title(f"Signal = {classes[int(np.round(pred))]}")
            plt.xlabel("Time (num. points)")
            plt.ylabel("Amplitude")
            plt.show()


def main():
    """Main function to train Deep Learning VAD models."""
    parser = argparse.ArgumentParser(
        description="Train a CNN for Voice Activity Detection."
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        required=True,
        help="TFRecords dataset directory path.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Trained model directory path.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Pre-trained checkpoint directory path.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="train",
        choices=["train", "eval", "predict"],
        help="TF estimator mode.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet1d",
        choices=["resnet1d"],
        help="Model name to train.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=1024,
        help="Audio signal input size.",
    )
    parser.add_argument(
        "--batch-size",
        "-bs",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=20,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--n-filters",
        type=int,
        default=[32, 64, 128],
        nargs=3,
        help="Number of filters for Conv layers.",
    )
    parser.add_argument(
        "--n-kernels",
        type=int,
        default=[8, 5, 3],
        nargs=3,
        help="Number of kernels for Conv layers.",
    )
    parser.add_argument(
        "--n-fc-units",
        type=int,
        default=[2048, 2048],
        nargs=2,
        help="Number of units for FC layers.",
    )
    parser.add_argument(
        "--n-classes",
        "-n",
        type=int,
        default=1,
        choices=[1],
        help="Number of output classes.",
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=0.00001,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--subsample",
        action="store_true",
        default=False,
        help="Audio signal subsampling option.",
    )
    parser.add_argument(
        "--fake-input",
        action="store_true",
        default=False,
        help="Debug option to train on 1 batch.",
    )
    args = parser.parse_args()

    params = {
        "model": args.model,
        "input_size": args.input_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "n_cnn_filters": args.n_filters,
        "n_cnn_kernels": args.n_filters,
        "n_fc_units": args.n_fc_units,
        "n_classes": args.n_classes,
        "lr": args.learning_rate,
        "subsample": args.subsample,
    }

    train(
        params=params,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        ckpt=args.ckpt,
        mode=args.mode,
        fake_input=args.fake_input,
    )


if __name__ == "__main__":
    main()
