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


def main():
    """Main function to train Deep Learning VAD models."""
    parser = argparse.ArgumentParser(description="train CNN for VAD")
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="/home/filippo/datasets/LibriSpeech/tfrecords/",
        help="tf records data directory",
    )
    parser.add_argument(
        "--model-dir", type=str, default="", help="pretrained model directory"
    )
    parser.add_argument(
        "--ckpt", type=str, default="", help="pretrained checkpoint directory"
    )
    parser.add_argument(
        "--mode", "-m", type=str, default="train", help="train, eval or predict"
    )
    parser.add_argument("--model", type=str, default="resnet1d", help="model name")
    parser.add_argument(
        "--input-size", type=int, default=1024, help="signal input size"
    )
    parser.add_argument("--batch-size", "-bs", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="train epochs")
    parser.add_argument("--n-filters", type=str, default="32-64-128")
    parser.add_argument("--n-kernels", type=str, default="8-5-3")
    parser.add_argument("--n-fc-units", type=str, default="2048-2048")
    parser.add_argument(
        "--n-classes",
        "-n",
        type=int,
        default=1,
        help="number of classes in output",
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=0.00001,
        help="initial learning rate",
    )
    parser.add_argument(
        "--fake-input",
        action="store_true",
        default=False,
        help="debug with 1 batch training",
    )
    parser.add_argument(
        "--subsample",
        action="store_true",
        default=False,
        help="subsample signal",
    )
    args = parser.parse_args()

    assert args.model in ["resnet1d"], "Wrong model name"
    assert len(args.n_filters.split("-")) == 3, "3 values required for --n-filters"
    assert len(args.n_kernels.split("-")) == 3, "3 values required for --n-kernels"
    assert len(args.n_fc_units.split("-")) == 2, "2 values required --n-fc-units"

    tfrecords_train = glob.glob(f"{args.data_dir}train/*.tfrecord")
    tfrecords_val = glob.glob(f"{args.data_dir}val/*.tfrecord")
    tfrecords_test = glob.glob(f"{args.data_dir}test/*.tfrecord")

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    logger = logging.getLogger(__name__)
    np.random.seed(0)
    tf.set_random_seed(0)
    tf.logging.set_verbosity(tf.logging.INFO)

    if not args.model_dir:
        save_dir = f"{args.data_dir}models/{args.model}/{datetime.now().isoformat()}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = args.model_dir

    params = {
        "model": args.model,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "n_cnn_filters": [int(x) for x in args.n_filters.split("-")],
        "n_cnn_kernels": [int(x) for x in args.n_kernels.split("-")],
        "n_fc_units": [int(x) for x in args.n_fc_units.split("-")],
        "n_classes": args.n_classes,
        "lr": args.learning_rate,
    }

    train_config = tf.estimator.RunConfig(
        save_summary_steps=10,
        save_checkpoints_steps=500,
        keep_checkpoint_max=20,
        log_step_count_steps=10,
    )

    ws = None
    if args.ckpt:
        ws = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=args.ckpt, vars_to_warm_start=".*"
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
    mode = mode_keys[args.mode]

    # Training & Evaluation on Train / Val set
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_input_fn = data_input_fn(
            tfrecords_train,
            batch_size=params["batch_size"],
            epochs=1,
            input_size=args.input_size,
            n_classes=params["n_classes"],
            subsample=args.subsample,
            shuffle=True,
            fake_input=args.fake_input,
        )
        eval_input_fn = data_input_fn(
            tfrecords_val,
            batch_size=params["batch_size"],
            epochs=1,
            input_size=args.input_size,
            n_classes=params["n_classes"],
            subsample=args.subsample,
            shuffle=False,
            fake_input=args.fake_input,
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
            input_size=args.input_size,
            n_classes=params["n_classes"],
            subsample=args.subsample,
            shuffle=False,
            fake_input=args.fake_input,
        )

        logger.info("Evaluation of test set ...")
        estimator.evaluate(input_fn=test_input_fn)

    # Prediction visualization on Test set
    elif mode == tf.estimator.ModeKeys.PREDICT:
        test_input_fn = data_input_fn(
            tfrecords_test,
            batch_size=params["batch_size"],
            epochs=1,
            input_size=args.input_size,
            n_classes=params["n_classes"],
            subsample=args.subsample,
            shuffle=False,
            fake_input=args.fake_input,
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


if __name__ == "__main__":
    main()
