# -*- coding: utf-8 -*-
"""
GCN Model
"""
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
from utils import get_graph, loss, grad, test

class GraphConvolutionLayer(keras.layers.Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self, input_dim, output_dim, support=1,
                 activation = None,   ## 在这个地方改output的激活函数
                 use_bias = True,
                 kernel_initializer = 'glorot_uniform',
                 bias_initializer = 'zeros',
                 kernel_regularizer = None,
                 bias_regularizer = None):
        super(GraphConvolutionLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activation = activation

    def build(self, nodes_shape):
        self.kernel = self.add_weight(shape = (self.input_dim, self.output_dim),
                                      initializer = self.kernel_initializer,
                                      name = 'kernel',
                                      regularizer = self.kernel_regularizer)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim, ),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer = self.bias_regularizer)
        else:
            self.bias = None
            
        self.built = True

    def call(self, nodes, edges):
        support = tf.matmul(nodes, self.kernel)

        output = tf.matmul(edges, support)

        if self.use_bias:
            output += self.bias

        if self.activation is not None:
            output = self.activation(output)
            
        return output
    
class GraphConvolutionModel(keras.Model):
    def __init__(self):
        super(GraphConvolutionModel, self).__init__()

        self.graph_conv_1 = GraphConvolutionLayer(9, 128,
                    activation = tf.keras.activations.relu,
                    kernel_regularizer = tf.keras.regularizers.l2(0.01))

        # self.graph_conv_2 = GraphConvolutionLayer(256, 64,
        #             activation=tf.keras.activations.relu,
        #             kernel_regularizer=tf.keras.regularizers.l2(0.01))

        # self.graph_conv_3 = GraphConvolutionLayer(64, 7, activation = tf.sigmoid)
        self.graph_conv_3 = GraphConvolutionLayer(128, 7)  ## TODO 做多分类不需要sigmoid，softmax激活选一个就行，因为只是预测成一类的

    def call(self, x, training=False):

        nodes = x[0]
        edges = x[1]
        h = self.graph_conv_1(nodes, edges)
        # h = self.graph_conv_2(h, edges)
        logits = self.graph_conv_3(h, edges)

        return logits
    


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    # path = './data/segmentation/segmentation_partial.csv'
    # data = pd.read_csv(path)
    # data = data.sample(frac = 1, random_state = 100)
    #
    # features = data.iloc[:,1:-2].values
    # Truth_labels = data['Label'].values.astype('int')
    # Candidate_labels = pd.concat([data['Label'], data.iloc[:, -2:]], axis = 1).values.astype('int')
    # # print(Candidate_labels)
    # adj = get_graph(Truth_labels[:1500], Candidate_labels)
    # print(np.sum(adj, 1))
    # graph = [features, adj]
    #
    # train_index = np.arange(1000)
    # test_index = np.arange(1000, 2308)
    #
    # train_mask = np.zeros(features.shape[0], dtype = np.bool)
    # test_mask = np.zeros(features.shape[0], dtype = np.bool)
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


    model = GraphConvolutionModel()
    # print(model(graph))
    optimizer = tf.keras.optimizers.Adam(learning_rate = 5e-4, decay = 1e-4)

    # 记录过程值，以便最后可视化
    train_loss_results = []
    train_accuracy_results = []
    train_test_results = []


    num_epochs = 10000
    labels = tf.one_hot(Truth_labels, depth = 7)

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
    # plt.savefig('./result/GCN_result.png')
    plt.savefig('./result/GCN_result_glass.png')
    plt.show()

    
    