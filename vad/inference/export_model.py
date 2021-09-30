"""Main entrypoint to export a TF trained model to inference format."""
import argparse
import logging
import sys

import numpy as np
import tensorflow as tf

from vad.training.estimator import VadEstimator
from vad.training.input_pipeline import FEAT_SIZE


def export_model(params, model_dir):
    """Export a trained TF model for inference format.

    Args:
        params (dict): dictionary of model training parameters
        model_dir (str, optional): path to pre-trained model directory. Defaults to None
    """
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    logger = logging.getLogger(__name__)
    np.random.seed(0)
    tf.set_random_seed(0)
    tf.logging.set_verbosity(tf.logging.INFO)

    train_config = tf.estimator.RunConfig(
        save_summary_steps=10,
        save_checkpoints_steps=1000,
        keep_checkpoint_max=20,
        log_step_count_steps=10,
    )

    estimator_obj = VadEstimator(params)
    estimator = tf.estimator.Estimator(
        model_fn=estimator_obj.model_fn,
        model_dir=model_dir,
        config=train_config,
        params=params,
    )

    feature_spec = {
        "features_input": tf.placeholder(
            dtype=tf.float32,
            shape=[1, FEAT_SIZE[1], FEAT_SIZE[0]],
        )
    }

    logger.info("Exporting TensorFlow trained model ...")
    raw_serving_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        feature_spec,
        default_batch_size=1,
    )
    estimator.export_savedmodel(model_dir, raw_serving_fn, strip_default_attrs=True)


def main():
    """Main function to export a TF trained model to inference format."""
    parser = argparse.ArgumentParser(
        description="Export a trained TF CNN model for inference."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Pre-trained model directory path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet1d",
        choices=["resnet1d"],
        help="Model name to train.",
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
    args = parser.parse_args()

    params = {
        "model": args.model,
        "n_cnn_filters": args.n_filters,
        "n_cnn_kernels": args.n_filters,
        "n_fc_units": args.n_fc_units,
        "n_classes": args.n_classes,
    }

    export_model(params=params, model_dir=args.model_dir)


if __name__ == "__main__":
    main()
