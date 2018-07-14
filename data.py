import tensorflow as tf
from utils import get_data
import numpy as np

(train_data, train_labels), (eval_data, eval_labels) = get_data()

train_input_func = tf.estimator.inputs.numpy_input_fn(
    x={"image": np.reshape(train_data, [-1, 28, 28, 1])},
    y=train_labels.astype(np.int64),
    batch_size=32,
    num_epochs=None,
    shuffle=True)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image": np.reshape(eval_data, [-1, 28, 28, 1])},
    y=eval_labels.astype(np.int64),
    num_epochs=1,
    shuffle=False)

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'image': np.reshape(eval_data, [-1, 28, 28, 1])},
    num_epochs=1,
    shuffle=False)

pred_labels = eval_labels