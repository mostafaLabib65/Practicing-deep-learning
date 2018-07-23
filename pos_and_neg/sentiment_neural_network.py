import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pos_and_neg.get_features import create_data


train_x, train_y, test_x, test_y = create_data('pos.txt', 'neg.txt')
n_nodes_layers = [len(train_x[0]),500,500,500]

n_classes = 2
batch_size = 100

x = tf.placeholder('float',[None, len(train_x[0])])
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_layers = [tf.Variable] * 3
    for i in range(3):
        hidden_layers[i] = {'weights': tf.Variable(tf.random_normal([n_nodes_layers[i], n_nodes_layers[i+1]])), 'biases': tf.Variable(tf.random_normal([n_nodes_layers[i+1]]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_layers[3], n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}


    layers = [tf.Variable]*3
    layers[0] = tf.add(tf.matmul(data, hidden_layers[0]['weights']), hidden_layers[0]['biases'])
    layers[0] = tf.nn.relu(layers[0])

    for i in range(1,3):
        layers[i] = tf.add(tf.matmul(layers[i-1], hidden_layers[i]['weights']), hidden_layers[i]['biases'])
        layers[i] = tf.nn.relu(layers[i])
    output = tf.add(tf.matmul(layers[2], output_layer['weights']), output_layer['biases'])

    return output



def train_neural_network(x,y):
    pridection = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pridection,labels=y))
    optimizor = tf.train.AdamOptimizer().minimize(cost)

    epochs = 6
    with tf.Session() as sess:
        costs = [0] * 6
        sess.run(tf.initialize_all_variables())
        for epoch in range(epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i+batch_size
                i = end
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizor, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                costs[epoch] = epoch_loss
            print('Epoch ', epoch+1, 'completed out of', epochs, 'loss ', epoch_loss)
        correct = tf.equal(tf.argmax(pridection, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x:test_x, y: test_y}))
        plt.plot([1, 2, 3, 4, 5, 6], costs)
        plt.show()



train_neural_network(x,y)