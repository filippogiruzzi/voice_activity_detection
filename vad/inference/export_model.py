import argparse
import logging
import sys

import numpy as np
import tensorflow as tf

from vad.training.estimator import VadEstimator
from vad.training.input_pipeline import FEAT_SIZE


def main():
    parser = argparse.ArgumentParser(
        description="export trained TensorFlow model for inference"
    )
    parser.add_argument(
        "--model-dir", type=str, default="", help="pretrained model directory"
    )
    parser.add_argument(
        "--ckpt", type=str, default="", help="pretrained checkpoint directory"
    )
    parser.add_argument("--model", type=str, default="resnet1d", help="model name")
    parser.add_argument("--n-filters", type=str, default="32-64-128")
    parser.add_argument("--n-kernels", type=str, default="8-5-3")
    parser.add_argument("--n-fc-units", type=str, default="2048-2048")
    parser.add_argument(
        "--n-classes", "-n", type=int, default=1, help="number of classes"
    )
    args = parser.parse_args()

    assert args.model in ["resnet1d"], "Wrong model name"
    assert len(args.n_filters.split("-")) == 3, "3 values required for --n-filters"
    assert len(args.n_kernels.split("-")) == 3, "3 values required for --n-kernels"
    assert len(args.n_fc_units.split("-")) == 2, "2 values required --n-fc-units"

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    logger = logging.getLogger(__name__)
    np.random.seed(0)
    tf.set_random_seed(0)
    tf.logging.set_verbosity(tf.logging.INFO)

    save_dir = args.model_dir

    np.random.seed(0)
    tf.set_random_seed(0)

    params = {
        "model": args.model,
        "n_classes": args.n_classes,
        "n_cnn_filters": [int(x) for x in args.n_filters.split("-")],
        "n_cnn_kernels": [int(x) for x in args.n_kernels.split("-")],
        "n_fc_units": [int(x) for x in args.n_fc_units.split("-")],
    }

    train_config = tf.estimator.RunConfig(
        save_summary_steps=10,
        save_checkpoints_steps=1000,
        keep_checkpoint_max=20,
        log_step_count_steps=10,
    )

    estimator_obj = VadEstimator(params)
    estimator = tf.estimator.Estimator(
        model_fn=estimator_obj.model_fn,
        model_dir=save_dir,
        config=train_config,
        params=params,
    )

    feature_spec = {
        "features_input": tf.placeholder(
            dtype=tf.float32, shape=[1, FEAT_SIZE[1], FEAT_SIZE[0]]
        )
    }

    logger.info("Exporting TensorFlow trained model ...")
    raw_serving_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        feature_spec, default_batch_size=1
    )
    estimator.export_savedmodel(save_dir, raw_serving_fn, strip_default_attrs=True)


if __name__ == "__main__":
    main()
