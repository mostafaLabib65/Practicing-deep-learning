import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.examples.tutorials.mnist import input_data




mnist = input_data.read_data_sets("tmp/data", one_hot = True)


epochs = 10
n_classes = 10
batch_size = 128
chunk = 28
n_chunks = 28
rnn_size = 128

x = tf.placeholder('float', [None, n_chunks,chunk])
y = tf.placeholder('float')


def rnn_model(data):
    layer = {'Weights': tf.Variable(tf.random_normal([rnn_size, n_classes])), 'biases': tf.Variable(tf.zeros([n_classes]))}

    data = tf.transpose(data, [1, 0, 2])
    data = tf.reshape(data, [-1, chunk])
    data = tf.split(data, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, data, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['Weights']) + layer['biases']

    return output


def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt


def train_neural_network(x, y):
    pridection = rnn_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pridection,labels=y))
    optimizor = tf.train.AdamOptimizer().minimize(cost)


    with tf.Session() as sess:
        costs = [0] * 10
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk))
                _, c = sess.run([optimizor, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                costs[epoch] = epoch_loss
            print('Epoch ', epoch+1, 'completed out of', epochs, 'loss ', epoch_loss)
        correct = tf.equal(tf.argmax(pridection, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk)), y: mnist.test.labels}))
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