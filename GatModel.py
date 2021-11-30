import tensorflow as tf
from tensorflow import keras
from utils import get_graph, loss, grad, test
from tqdm import tqdm
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU

class GraphAttention(Layer):

    def __init__(self,
                 F_,
                 attn_heads = 4,
                 attn_heads_reduction = 'average',  # {'concat', 'average'}
                 dropout_rate = 0.5,
                 activation = None,
                 use_bias = True,
                 kernel_initializer = 'glorot_uniform',
                 bias_initializer = 'zeros',
                 attn_kernel_initializer = 'glorot_uniform',
                 kernel_regularizer = keras.regularizers.l2(0.001),
                 bias_regularizer = None,
                 attn_kernel_regularizer = None,
                 activity_regularizer = None,
                 kernel_constraint = None,
                 bias_constraint = None,
                 attn_kernel_constraint = None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head),)
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (N x F)
        A = inputs[1]  # Adjacency matrix (N x N)

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(features, attention_kernel[0])    # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(features, attention_kernel[1])  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            dense = attn_for_self + K.transpose(attn_for_neighs)  # (N x N) via broadcasting

            # Add nonlinearty
            dense = LeakyReLU(alpha=0.2)(dense)

            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e9 * (1.0 - A)
            dense += mask

            # Apply softmax to get attention coefficients
            dense = K.softmax(dense)  # (N x N)

            # Apply dropout to features and attention coefficients
            dropout_attn = Dropout(self.dropout_rate)(dense)  # (N x N)
            dropout_feat = Dropout(self.dropout_rate)(features)  # (N x F')

            # Linear combination with neighbors' features
            node_features = K.dot(dropout_attn, dropout_feat)  # (N x F')

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')

        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape


class GraphAttention_Model(keras.Model):
    def __init__(self):
        super(GraphAttention_Model, self).__init__()
        self.gat_attention_1 = GraphAttention(7, activation = 'relu')
        self.gat_attention_2 = GraphAttention(7)

    def call(self, x, training=False):
        nodes = x[0]
        edges = x[1]
        h = self.gat_attention_1(x)
        logits = self.gat_attention_2([h, edges])
        return logits


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # path = './data/segmentation/segmentation_partial.csv'
    # data = pd.read_csv(path)
    # data = data.sample(frac=1, random_state=100)
    #
    # features = data.iloc[:, 1:-2].values
    # Truth_labels = data['Label'].values.astype('int')
    # Candidate_labels = pd.concat([data['Label'], data.iloc[:, -2:]], axis=1).values.astype('int')
    # adj = get_graph(Truth_labels[:1500], Candidate_labels)
    # graph = [features, adj]
    #
    # train_index = np.arange(1000)
    # test_index = np.arange(1000, 2308)
    #
    # train_mask = np.zeros(features.shape[0], dtype=np.bool)
    # test_mask = np.zeros(features.shape[0], dtype=np.bool)
    #
    # train_mask[train_index] = True
    # test_mask[test_index] = True

    ########################################### glass ###############################################
    path = './data/glass/glass_partial.csv'
    data = pd.read_csv(path)
    data = data.sample(frac=1, random_state=100)
    features = data.iloc[:, 1:-3].values
    Truth_labels = data['Truth label'].values.astype('int')
    Candidate_labels = pd.concat([data['Truth label'], data.iloc[:, -2:]], axis=1).values.astype('int')
    adj = get_graph(Truth_labels[:200], Candidate_labels)
    graph = [features, adj]

    train_index = np.arange(100)
    test_index = np.arange(100, 214)
    train_mask = np.zeros(features.shape[0], dtype=np.bool)
    test_mask = np.zeros(features.shape[0], dtype=np.bool)
    train_mask[train_index] = True
    test_mask[test_index] = True


    # model = GraphAttention(7, activation = 'sigmoid')
    model = GraphAttention_Model() ## TODO 同理一样的道理，最后不需要sigmoid激活

    # print(model(graph))
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, decay=1e-4)

    # 记录过程值，以便最后可视化
    train_loss_results = []
    train_accuracy_results = []
    train_test_results = []
    num_epochs = 10000
    labels = tf.one_hot(Truth_labels, depth=7)

    for epoch in tqdm(range(num_epochs)):
        loss_value, grads = grad(model, graph, labels, train_mask)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_acc = test(model, graph, train_mask, labels)
        test_acc = test(model, graph, test_mask, labels)

        train_loss_results.append(loss_value.numpy())
        train_accuracy_results.append(train_acc.numpy())
        train_test_results.append(test_acc.numpy())

    fig, axes = plt.subplots(3, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].plot(train_accuracy_results)

    axes[2].set_ylabel("Test Acc", fontsize=14)
    axes[2].plot(train_test_results)
    plt.savefig('./result/GAT_result.png')
    plt.show()