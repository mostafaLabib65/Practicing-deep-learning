
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


lemmetizer = WordNetLemmatizer()


train_x = [0] * 2638
n_nodes_layers = [len(train_x),500,500,500]

n_classes = 2

batch_size = 32
total_batches = int(1600000/batch_size)
hm_epochs = 1

hm_data = 200000

x = tf.placeholder('float',[None, len(train_x)])
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_layers = [tf.Variable] * (len(n_nodes_layers)-1)
    for i in range((len(n_nodes_layers)-1)):
        hidden_layers[i] = {'weights': tf.Variable(tf.random_normal([n_nodes_layers[i], n_nodes_layers[i+1]])), 'biases': tf.Variable(tf.random_normal([n_nodes_layers[i+1]]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_layers[(len(n_nodes_layers)-1)], n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}


    layers = [tf.Variable]*(len(n_nodes_layers)-1)
    layers[0] = tf.add(tf.matmul(data, hidden_layers[0]['weights']), hidden_layers[0]['biases'])
    layers[0] = tf.nn.relu(layers[0])

    for i in range(1,(len(n_nodes_layers)-1)):
        layers[i] = tf.add(tf.matmul(layers[i-1], hidden_layers[i]['weights']), hidden_layers[i]['biases'])
        layers[i] = tf.nn.relu(layers[i])
    output = tf.add(tf.matmul(layers[len(layers)-1], output_layer['weights']), output_layer['biases'])

    return output



def train_neural_network(x,y):

    tf_log = 'tf_log'
    pridection = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pridection,labels=y))
    optimizor = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        costs = [0] * 6
        sess.run(tf.initialize_all_variables())
        try:
            epoch = int(open(tf_log, 'r').read().split('\n')[-2]) +1
            print('Starting epoch: ', epoch)
        except Exception as e:
            epoch = 1
        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess, "model.ckpt")
            epoch_loss = 1
            with open('lexicon.pickle', 'rb') as f:
                counter = 0
                lexicon = pickle.load(f)
            with open('train_set_shuffled.csv',buffering=200000, encoding='latin-1') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
                    counter +=1
                    label = line.split(':::')[0]
                    tweet = line.split(':::')[1]
                    words = word_tokenize(tweet.lower())
                    words = [lemmetizer.lemmatize(i) for i in words]
                    features = np.zeros(len(lexicon))
                    for word in words:
                        if word in lexicon:
                            index = lexicon.index(word)
                            features[index] +=1
                    line_x = list(features)
                    line_y = eval(label)
                    batch_x.append(line_x)
                    batch_y.append(line_y)
                    if len(batch_x) >= batch_size:
                        _, c = sess.run([optimizor, cost], feed_dict={x: batch_x, y: batch_y})
                        epoch_loss += c
                        costs[epoch] = epoch_loss
                        batch_x = []
                        batch_y = []
                        batches_run += 1
                        print('Batch run:', batches_run, '/', total_batches, '| Epoch:', epoch, '| Batch Loss:', c, )

            saver.save(sess, "model.ckpt")
            print('Epoch ', epoch+1, 'completed out of', hm_epochs, 'loss ', epoch_loss)
            with open(tf_log,'a') as f:
                f.write(str(epoch)+'\n')
        # plt.plot([i for i in range(hm_epochs)], costs)
        # plt.show()



def test_nn():
    saver = tf.train.Saver()
    pridection = neural_network_model(x)
    with tf.Session as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            try:
                saver.restore(sess, "model.ckpt")
            except Exception as e:
                print(str(e))
            e_loss = 0
    correct = tf.equal(tf.argmax(pridection, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    feature_sets = []
    labels = []
    counter = 0
    with open('processed-test-set.csv', buffering=200000, encoding='latin-1') as f:
        for line in f:
            try:
                features = list(eval(line.split(':::')[0]))
                label = list(eval(line.split(':::')[1]))
                feature_sets.append(features)
                labels.append(label)
                counter += 1
            except Exception as e:
                pass
    print("Tested: ", counter, " sample")
    test_x = np.array(feature_sets)
    test_y = np.array(labels)
    print('Accuracy: ', accuracy.eval({x: test_x, y: test_y}))


def use_neural_network(input_data):
    saver = tf.train.Saver()
    prediction = neural_network_model(x)
    with open('lexicon.pickle', 'rb') as f:
        lexicon = pickle.load(f)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, "model.ckpt")
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmetizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                # OR DO +=1, test both
                features[index_value] += 1

        features = np.array(list(features))
        # pos: [1,0] , argmax: 0
        # neg: [0,1] , argmax: 1
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [features]}), 1)))
        if result[0] == 0:
            print('Positive:', input_data)
        elif result[0] == 1:
            print('Negative:', input_data)



train_neural_network(x,y)