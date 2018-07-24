# you can download the dataset from http://help.sentiment140.com/for-students/

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
import pandas as pd

lemmetizer = WordNetLemmatizer()


def init_processing(fin, fout):
    output = open(fout, 'a')
    with open(fin, buffering=200000, encoding='latin-1') as file:
        try:
            for line in file:
                line = line.replace('"', '')
                polarity = line.split(',')[0]
                if polarity == '0':
                    polarity = [1, 0]
                elif polarity == '4':
                    polarity = [0, 1]
                else:
                    continue
                tweet = line.split(',')[-1]
                outline = str(polarity) + ':::' + tweet
                output.write(outline)
        except Exception as e:
            print(str(e))
    output.close()


def create_lexicon(input_file):
    lexicon = []
    with open(input_file, 'r', buffering=100000, encoding='latin-1') as f:
        try:
            counter = 1
            content = ''
            for line in f:
                counter += 1
                if (counter / 2500.0).is_integer():
                    tweet = line.split(':::')[1]
                    content += ' ' + tweet
                    words = word_tokenize(content)
                    words = [lemmetizer.lemmatize(i) for i in words]
                    lexicon = list(set(lexicon + words))
                    print(counter, len(lexicon))

        except Exception as e:
            print(str(e))

    with open('lexicon.pickle', 'wb') as f:
        pickle.dump(lexicon, f)


def shuffle_data(fin):
    df = pd.read_csv(fin, error_bad_lines=False)
    df = df.iloc[np.random.permutation(len(df))]
    print(df.head())
    df.to_csv('train_set_shuffled.csv', index=False)


def convert_to_vec(fin, fout, lexicon_pickle):
    with open(lexicon_pickle, 'rb') as f:
        lexicon = pickle.load(f)
    outfile = open(fout, 'a')
    with open(fin, buffering=20000, encoding='latin-1') as f:
        counter = 0
        for line in f:
            counter += 1
            label = line.split(':::')[0]
            tweet = line.split(':::')[1]
            current_words = word_tokenize(tweet.lower())
            current_words = [lemmetizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            outline = str(features) + '::' + str(label) + '\n'
            outfile.write(outline)
        print(counter)


def create_test_data_pickle(fin):
    feature_sets = []
    labels = []
    counter = 0
    with open(fin, buffering=20000) as f:
        for line in f:
            try:
                features = list(eval(line.split('::')[0]))
                label = list(eval(line.split('::')[1]))
                feature_sets.append(features)
                labels.append(label)
                counter += 1
            except:
                pass
    print(counter)
    feature_sets = np.array(feature_sets)
    labels = np.array(labels)


init_processing('training.1600000.processed.noemoticon.csv','train_set.csv')
init_processing('testdata.manual.2009.06.14.csv','test_set.csv')
create_lexicon('train_set.csv')
convert_to_vec('test_set.csv','processed-test-set.csv','lexicon.pickle')
shuffle_data('train_set.csv')
create_test_data_pickle('processed-test-set.csv')