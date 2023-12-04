NAME = "flax-recommender-system"

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from flax.training.train_state import TrainState
from orbax.export import ExportManager, JaxModule, ServingConfig


def init_model(input_shape: Tuple, model: nn.Module):
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones(input_shape))
    return params


def create_train_state(model, params) -> TrainState:
    lr = 0.001
    momentum = 0.9
    tx = optax.sgd(lr, momentum)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def save(state, preprocessing_fn, output_dir, etr=None):
    # Construct a JaxModule where JAX->TF conversion happens.
    jax_module = JaxModule(state.params, state.apply_fn)
    # Export the JaxModule along with one or more serving configs.
    export_mgr = ExportManager(
        jax_module,
        [
            ServingConfig(
                "serving_default",
                # input_signature=[tf.TensorSpec(shape=(10), dtype=tf.float32)],
                tf_preprocessor=preprocessing_fn,
                # tf_postprocessor=example1_postprocess
                extra_trackable_resources=etr,
            ),
        ],
    )
    export_mgr.save(output_dir)


def load(output_dir):
    loaded_model = tf.saved_model.load(output_dir)
    return loaded_model


def inference(loaded_model, inputs):
    loaded_model_outputs = loaded_model(inputs)
    return loaded_model_outputs
