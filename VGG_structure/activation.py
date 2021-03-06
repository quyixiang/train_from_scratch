import enum
import tensorflow as tf

class Activation(enum.Enum):
    identity = tf.identity
    relu = tf.nn.relu
    softmax = tf.nn.softmax
