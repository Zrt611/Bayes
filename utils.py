import numpy as np
import tensorflow as tf
from tensorflow import keras

def get_graph(Truth_labellst, Candidate_labellst):
    length = Candidate_labellst.shape[0]
    #print(labellst)
    graph = np.zeros([length, length])
    for i in range(length):
        for j in range(length):
            if len(set(Candidate_labellst[i]) & set(Candidate_labellst[j])) >= 2:
                graph[i, j] = 1
            else:
                graph[i, j] = 0
    Truth_length = len(Truth_labellst)
    graph[:Truth_length, :Truth_length] = 0
    for i in range(Truth_length):
        for j in range(Truth_length):
            if Truth_labellst[i] == Truth_labellst[j]:
                graph[i, j] = 1
            else:
                graph[i, j] = 0
    return graph

loss_object = keras.losses.CategoricalCrossentropy(from_logits = True)

## compute loss
def loss(model, x, y, train_mask, training):
    y_ = model(x, training=training)
    test_mask_logits = tf.gather_nd(y_, tf.where(train_mask))
    masked_labels = tf.gather_nd(y, tf.where(train_mask))

    return loss_object(y_true=masked_labels, y_pred=test_mask_logits)


## compute grad
def grad(model, inputs, targets, train_mask):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, train_mask, training=True)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def test(model, graph, mask, labels):
    logits = model(graph)

    test_mask_logits = tf.gather_nd(logits, tf.where(mask))
    masked_labels = tf.gather_nd(labels, tf.where(mask))
    # print(tf.math.argmax(test_mask_logits, -1))

    ll = tf.math.equal(tf.math.argmax(masked_labels, -1), tf.math.argmax(test_mask_logits, -1))
    accuarcy = tf.reduce_mean(tf.cast(ll, dtype=tf.float64))

    return accuarcy
