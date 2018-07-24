import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter


lemmetizer = WordNetLemmatizer()
n_lines = 10000000


def create_lexicon(pos, neg):
    lexicon = []
    for file in [pos, neg]:
        with open(file, 'r') as f:
            content = f.readlines()
            for l in content[:n_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
    lexicon = [lemmetizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    output = []

    for word in w_counts:
        if 50 < w_counts[word] < 1000:
            output.append(word)
    print(len(output))
    return output


def get_features(input, lexicon, classification):
    features_set = []
    with open(input, 'r') as f:
        lines = f.readlines()
        for l in lines[:n_lines]:
            words = word_tokenize(l.lower())
            words = [lemmetizer.lemmatize(i) for i in words]
            features = np.zeros(len(lexicon))
            for w in words:
                if w.lower() in lexicon:
                    index = lexicon.index(w.lower())
                    features[index] += 1

            features = list(features)
            features_set.append([features, classification])
    return features_set


def create_data(pos, neg, test_ratio=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += get_features('pos.txt', lexicon, [1, 0])
    features += get_features('neg.txt', lexicon, [0, 1])
    random.shuffle(features)
    features = np.array(features)
    testing_size = int(test_ratio*len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])

    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_data('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)














