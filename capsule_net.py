import tensorflow as tf
from utils import *

def primary_caps_layer(in_tensor, caps_size, n_maps, n_dims, conv_params):
    conv1 = in_tensor
    for conv_paramsi in conv_params:
        conv1 = tf.layers.conv2d(conv1, **conv_paramsi)

    caps_raw = tf.reshape(conv1, [-1, n_maps * caps_size * caps_size, n_dims])

    return squash(caps_raw)

def routing_by_agreement(raw_weights, caps_predicted, agreement=None):

    if agreement is not None:
        raw_weights = tf.add(raw_weights, agreement)

    routing_weights = tf.nn.softmax(raw_weights, axis=2)

    weighted_preds = tf.multiply(routing_weights, caps_predicted)
    weighted_sum = tf.reduce_sum(weighted_preds, axis=1, keepdims=True)
    return raw_weights, weighted_sum

def model_fn(features, labels, mode, params):

    n_maps1 = 32
    n_dims1 = 8
    caps_size1 = 6
    n_caps1 = n_maps1 * caps_size1 * caps_size1

    n_caps2 = 10
    n_dims2 = 16

    conv_params = [{
        'filters': 256,
        'kernel_size': 9,
        'strides': 1,
        'padding': 'valid',
        'activation': tf.nn.relu
    },
    {
        'filters': n_maps1 * n_dims1,
        'kernel_size': 9,
        'strides': 2,
        'padding': 'valid',
        'activation': tf.nn.relu
    }]

    # 1st Capsule layer
    caps1_out = primary_caps_layer(features['image'], caps_size1, n_maps1, n_dims1, conv_params)

    W_trans_tiled = init_trans_matrix(n_caps1, n_caps2, n_dims1, n_dims2, tf.shape(features['image'])[0])
    caps1_out_tiled = get_tiled_caps(caps1_out, n_caps2)

    # Prediction for 2nd layer
    caps2_predicted = tf.matmul(W_trans_tiled, caps1_out_tiled)

    # Routing by agreement
    if params['rounds'] == 2:
        ## Let's see 2 rounds only
        ## Round 1
        batch_size = tf.shape(caps2_predicted)[0]
        raw_weights = tf.zeros([batch_size, n_caps1, n_caps2, 1, 1], dtype=tf.float32)

        raw_weights, weighted_sum = routing_by_agreement(raw_weights, caps2_predicted)
        caps2_output_round_1 = squash(weighted_sum, axis=-2)

        caps2_output_round_1_tiled = tf.tile(caps2_output_round_1, [1, n_caps1, 1, 1, 1])
        agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled, transpose_a=True)

        ## Round 2
        raw_weights, weighted_sum_2 = routing_by_agreement(raw_weights, caps2_predicted, agreement)
        caps2_output = squash(weighted_sum_2, axis=-2)
    else:
        ## In Case of n Rounds
        ### Round 1
        batch_size = tf.shape(caps2_predicted)[0]
        raw_weights = tf.zeros([batch_size, n_caps1, n_caps2, 1, 1], dtype=tf.float32)

        weighted_sum = routing_by_agreement(raw_weights, caps2_predicted, n_caps1, n_caps2)
        caps2_output_round_1 = squash(weighted_sum, axis=-2)

        caps2_output_round_1_tiled = tf.tile(caps2_output_round_1, [1, n_caps1, 1, 1, 1])
        agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled, transpose_a=True)

        ### n-1 Rounds - dynamic loop
        counter = tf.constant(1)

        def condition(*args):
            return tf.less(args[0], params['rounds'])

        def loop_body(*args):
            """
            args[0] -> counter
            args[1] -> raw_weights
            args[2] -> caps2_predicted
            args[3] -> agreement
            args[4] -> caps2_output
            args[5] -> n_caps1
            """
            raw_weights, weighted_sum = routing_by_agreement(*args[1:4])
            caps2_output = squash(weighted_sum, axis=-2)

            caps2_output_round_n_tiled = tf.tile(caps2_output, [1, args[5], 1, 1, 1])
            agreement = tf.matmul(args[2], caps2_output_round_n_tiled, transpose_a=True)

            return tf.add(args[0], 1), raw_weights, args[2], agreement, caps2_output, args[5]

        (counter, raw_weights, 
         caps2_predicted, caps2_output,
         agreement, _) = tf.while_loop(condition, loop_body, [counter, raw_weights,
                                                              caps2_predicted, agreement,
                                                              caps2_output_round_1, n_caps1])

    y_probs = safe_norm(caps2_output, axis=-2)
    y_probs_argmax = tf.argmax(y_probs, axis=2)
    y_pred = tf.squeeze(y_probs_argmax, axis=[1,2])

    # Decoder
    ## Create Reconstruction Mask
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        reconstruction_targets = labels
    else:
        reconstruction_targets = y_pred

    reconstruction_mask = tf.one_hot(reconstruction_targets, depth=n_caps2)
    reconstruction_mask_reshaped = tf.reshape( reconstruction_mask, [-1, 1, n_caps2, 1, 1])

    ## Mask caps2_output
    caps2_output_masked = tf.multiply(caps2_output, reconstruction_mask_reshaped)
    decoder_in = tf.reshape(caps2_output_masked, [-1, n_caps2 * n_dims2])

    ## Feed through Decoder
    decoder_out = decoder(decoder_in, [512, 1024, 784])
    y_constructed = tf.reshape(decoder_out, [-1, 28, 28, 1])
    tf.summary.image('decoder_out', y_constructed)

    # Make Estimator Spec
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'reconstruction': y_constructed,
            'y': y_pred
        }
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
    else:
        # accuracy
        acc = accuracy(labels, y_pred)
        tf.summary.scalar('accuracy', acc)

        # Margin Loss
        ## Computing T -> One if present else 0 -> One hot vector for multi-class classification
        T = tf.one_hot(labels, depth=n_caps2)
        m_loss = margin_loss(y_probs, T, params['m_plus'], params['m_minus'])
        tf.summary.scalar('m_loss', m_loss)

        # Reconstruction loss
        r_loss = reconstruction_loss(features['image'], decoder_out)
        tf.summary.scalar('reconstruction_loss', r_loss)

        # Total loss
        loss = tf.add(m_loss, params['alpha'] * r_loss)

        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        spec= tf.estimator.EstimatorSpec(mode=mode,
                                         loss=loss, train_op=train_op)

    return spec
