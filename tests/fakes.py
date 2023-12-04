import tensorflow as tf
import flax.linen as nn


class TestLookupTablePreprocessing:
    def __init__(self, lookup_table):
        self.lookup_table = lookup_table

    @tf.function(input_signature=[{"x": tf.TensorSpec(shape=(2,), dtype=tf.string)}])
    def apply(self, inputs):
        return self.lookup_table(inputs["x"])


class TestFlaxAdditionModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x + 10
        return x
