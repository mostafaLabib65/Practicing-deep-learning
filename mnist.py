import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import time




mnist = input_data.read_data_sets("tmp/data", one_hot = True)


n_nodes_layers = [784,500,500,500]

n_classes = 10
batch_size = 100

x = tf.placeholder('float',[None, 784])
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


def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt


def train_neural_network(x,y):
    pridection = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pridection,labels=y))
    optimizor = tf.train.AdamOptimizer().minimize(cost)
    tf_x = tf.placeholder('float', [None, 784])
    tf_p = tf.placeholder('float')

    epochs = 6

    with tf.Session() as sess:
        costs = [0] * 6
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizor, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                costs[epoch] = epoch_loss
            print('Epoch ', epoch+1, 'completed out of', epochs, 'loss ', epoch_loss)
        correct = tf.equal(tf.argmax(pridection, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x:mnist.test.images, y: mnist.test.labels}))
        plt.plot([1, 2, 3, 4, 5, 6], costs)
        plt.show()
        test_x = mnist.test.images
        p = tf.argmax(pridection, 1)
        p = tf.cast(p, 'float')
        p = p.eval({x: test_x})
        while True:
            n = np.random.randint(0, p.shape[0])
            print('Model Prediction: ', p[n], '\n')
            print('---------------------\n')
            gen_image(test_x[n]).show()



train_neural_network(x,y)