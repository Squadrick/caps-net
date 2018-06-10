import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tf_util as U
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data")

ROUTE_ITER = 3

def capsule(inputs, units, dim):
    shape = inputs.get_shape() # [batch, units, dim]
    units_in = shape[1]
    dim_in = shape[2]
    b = tf.get_variable('b', shape=[1, units_in, units, 1, 1],
            dtype=tf.float32, initializer=tf.zeros_initializer)
    w = tf.get_variable('w', shape=[1, units_in, units, dim_in, dim],
            dtype=tf.float32, initializer=U.normc_initializer(0.1))
    inputs = tf.expand_dims(tf.expand_dims(inputs, axis=2), axis=2)
    inputs = tf.tile(inputs, [1, 1, units, 1, 1])
    w = tf.tile(w, [tf.shape(inputs)[0], 1, 1, 1, 1])
    inputs = tf.matmul(inputs, w)
    
    b = tf.tile(b, [tf.shape(inputs)[0], 1, 1, 1, 1])
    for i in range(ROUTE_ITER):
        c = tf.nn.softmax(b, dim=2)
        outputs = squash(U.sum(c * inputs, axis=1, keepdims=True))
        b += U.sum(inputs*outputs, axis=-1, keepdims=True)

    c = tf.nn.softmax(b, dim=2)
    outputs = squash(U.sum(c*inputs, axis=1, keepdims=True))
    outputs = tf.reshape(outputs, [-1, units, dim])
    return outputs

def capsule_conv(inputs, filters, dim, kernel_size, strides):
    shape = inputs.get_shape()
    dim_in = shape[-2].value
    filters_in = shape[-1].value
    w, h = kernel_size
    x, y = strides
    kernel = (w, h, dim_in)
    outputs = tf.layers.conv3d(inputs, filters*dim, (w, h, dim_in), (x, y, 1))
    outputs = squash(outputs, -2)
    return outputs 

def squash(sj, axis=-1):
    norm_sq = U.sum(tf.square(sj), axis = axis, keepdims=True)
    factor = norm_sq / ((1 + norm_sq) * tf.sqrt(norm_sq + 1e-9))
    return factor * sj

def margin_loss(labels, preds, upper_thres=0.9, lower_thres=0.1, lam = 0.5):
    present = labels * tf.square(tf.nn.relu(upper_thres - preds))
    absent  = (1 - labels) * tf.square(tf.nn.relu(preds - lower_thres))

    return U.mean(present + lam * absent, axis = -1)

def reconstruction_loss(true_img, pred_img, alpha = 0.005):
    return alpha * U.mean(tf.square(true_img - pred_img), axis = -1)

def reconstruct_fc(caps):
    with tf.variable_scope("reconstruction"):
        init = U.normc_initializer(0.1)
        fc1 = U.dense(caps, 512, name = "fc1", weight_init = init)
        fc1_act = tf.nn.relu(fc1)
        fc2 = U.dense(fc1_act, 1024, name = "fc2", weight_init = init)
        fc2_act = tf.nn.relu(fc2)
        fc3 = U.dense(fc2_act, 784, name = "fc3", weight_init = init)
        fc3_act = tf.nn.sigmoid(fc3)
        return fc3_act

def mask_scene(caps, mask):
    idx = tf.cast(tf.argmax(mask, axis=1), tf.int32)
    mask = tf.one_hot(indices=idx, depth=10)
    return tf.layers.flatten(caps*tf.expand_dims(mask, -1))

def caps_net(inputs, labels):
    inputs = tf.reshape(inputs, [-1, 28, 28, 1])

    conv1 = U.conv2d(inputs, 256, "conv1", filter_size=(3,3), stride=(1,1), pad="VALID")
    conv1_act = tf.nn.relu(conv1) # [-1, 20, 20, 256]
    conv1_act = tf.expand_dims(conv1_act, axis=-2) 
    primary_caps = capsule_conv(conv1_act, 32, 8, kernel_size=(9,9), strides=(2, 2))
    primary_caps = tf.reshape(primary_caps, [-1, primary_caps.shape[1].value*primary_caps.shape[2].value*32, 8])

    digitscaps = capsule(primary_caps, 10, 16)

    lengths = tf.sqrt(U.sum(tf.square(digitscaps),axis=2) + 1e-9)

    preds = tf.argmax(lengths, axis = -1)
    probs = tf.nn.softmax(lengths)

    masked_digitscaps = mask_scene(digitscaps, lengths)
    reconstruction_pred = reconstruct_fc(masked_digitscaps)

    r_loss = reconstruction_loss(tf.reshape(inputs, [-1, 784]), reconstruction_pred)
    m_loss = margin_loss(labels, lengths)
    loss = U.mean(m_loss + r_loss, axis=-1)
    opti = tf.train.AdamOptimizer()
    train = opti.minimize(loss)

    corr_pred = tf.equal(preds, tf.argmax(labels, 1))
    acc = U.mean(tf.cast(corr_pred, tf.float32))
    
    r_loss = U.mean(r_loss, axis=-1)
    m_loss = U.mean(m_loss, axis=-1)

    return train, acc, r_loss, m_loss, reconstruction_pred

x = tf.placeholder(shape=[None, 784], dtype=tf.float32)
y = tf.placeholder(shape=[None] , dtype=tf.float32)
y_oh = tf.one_hot(indices=tf.cast(y, tf.int32), depth=10)

sess = tf.Session()

train, acc, r_l, m_l, recon = caps_net(x, y_oh)

sess.run(tf.global_variables_initializer())
for i in range(1000):
    x_t, y_t = mnist.train.next_batch(1)

    _, a, l1, l2, r = sess.run([train, acc, r_l, m_l, recon], 
            feed_dict = {x: x_t, y:y_t})
    
    print("Loss:", l2,  "\t Acc: ", a)
    if i%25 == 0:
        x_t, y_t = mnist.test.next_batch(100)
        l1,l2,a,img = sess.run([r_l, m_l, acc, recon], feed_dict={x:x_t, y:y_t})
        print("Test Loss:", l2, "\tAcc:", a)
        print("LABEL =====>", y_t[0])
        ig = np.reshape(img[0], (28, 28))
        plt.imshow(ig)
        plt.show()
    
