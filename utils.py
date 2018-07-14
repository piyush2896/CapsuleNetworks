import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def squash(tensor, epsilon=1e-7, axis=-1):
    squared_norm = tf.reduce_sum(tf.square(tensor), axis, keepdims=True)
    scale = tf.divide(squared_norm, tf.add(1., squared_norm))
    unit_vectors = tf.divide(tensor, tf.sqrt(squared_norm+epsilon))

    return tf.multiply(scale, unit_vectors)

def init_trans_matrix(n_caps1, n_caps2, n_dims1, n_dims2, batch_size, init_sigma=0.1):

    W_init = tf.random_normal(
        shape=[1, n_caps1, n_caps2, n_dims2, n_dims1],
        stddev=init_sigma, dtype=tf.float32)
    W = tf.Variable(W_init)

    return tf.tile(W, [batch_size, 1, 1, 1, 1])

def get_tiled_caps(caps_out, n_caps):
    caps_out_expanded = tf.expand_dims(caps_out, -1)
    caps_out_tile = tf.expand_dims(caps_out_expanded, 2)
    return tf.tile(caps_out_tile, [1, 1, n_caps, 1, 1])

def safe_norm(s, axis=-1, epsilon=1e-7, keepdims=False):
    squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                 keepdims=keepdims)
    return tf.sqrt(squared_norm + epsilon)

def decoder(decoder_in, n_units):
    decoder_out = decoder_in
    for units in n_units:
        decoder_out = tf.layers.dense(decoder_out, units, activation=tf.nn.relu)
    return decoder_out

def margin_loss(y_probs, T, m_plus, m_minus, lambda_=0.5):
    present_error_raw = tf.square(tf.maximum(0., m_plus-y_probs))
    present_error = tf.reshape(present_error_raw, shape=(-1, 10))

    absent_error_raw = tf.square(tf.maximum(0., y_probs-m_minus))
    absent_error = tf.reshape(absent_error_raw, shape=(-1, 10))

    loss = tf.add(T * present_error, lambda_ * (1. - T) * absent_error)
    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

def reconstruction_loss(y, y_constructed):
    y = tf.reshape(y, [-1, 784])
    squared_diff = tf.square(y - y_constructed)
    return tf.reduce_mean(squared_diff)

def accuracy(y, y_pred):
    correct = tf.equal(y, y_pred)
    return tf.reduce_mean(tf.cast(correct, tf.float32))

def get_data():
    #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    mnist = input_data.read_data_sets("./mnist")
    train_imgs = mnist.train.images
    train_labels = mnist.train.labels
    eval_imgs = mnist.test.images
    eval_labels = mnist.test.labels
    return (train_imgs, train_labels), (eval_imgs, eval_labels)
