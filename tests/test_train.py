from trainer import train
from tests import fakes
import os
import jax.numpy as jnp
import tensorflow as tf


def test_name():
    assert train.NAME == "flax-recommender-system"


# leverage build in pytest fixture with tmp_path
def test_inference_exported_saved_model_with_preprocessing(tmpdir):
    # build preprocessing function
    input_shape = (2,)
    raw_inputs = {"x": ["a", "b"]}
    model = fakes.TestFlaxAdditionModel()
    vocab_src_dir = os.path.join(tmpdir, "vocab/")
    model_src_dir = os.path.join(tmpdir, "model/")

    # if we don't make the directory, we get a silent failure
    if not tf.io.gfile.exists(vocab_src_dir):
        tf.io.gfile.makedirs(vocab_src_dir)

    # build preprocessing layer
    lookup_layer = tf.keras.layers.StringLookup()
    lookup_layer.adapt(["a", "b", "c"])
    lookup_layer.save_assets(vocab_src_dir)

    # We have to explicitely make it trackable
    # and we want to save the preprocessing layer separately and load it
    new_lookup_layer = tf.keras.layers.StringLookup()
    new_lookup_layer.load_assets(vocab_src_dir)
    p = fakes.TestLookupTablePreprocessing(new_lookup_layer)

    # main model
    state = train.create_train_state(model, train.init_model(input_shape, model))

    # the etr (extra trackable resources) is a list of objects that should be tracked.
    # the lookup table is a trackable resource, but it is not an attribute of the
    # tf_preprocessor, so we have to add it to the etr list.
    train.save(state, p.apply, model_src_dir, etr=[p.lookup_table])

    # reload the model
    loaded = train.load(model_src_dir)

    # make sure we can make a prediction
    preds = train.inference(loaded, raw_inputs)
    assert preds.shape == (2,)
