from __future__ import absolute_import, division, print_function, unicode_literals

import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from keras_preprocessing.text import Tokenizer

train_file_path = "storyplot.csv"

docs=[]

def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(file_path,
        batch_size=1,
        label_name = 'storyid',
        na_value="",
        num_epochs=1,
        ignore_errors=True,
        shuffle=False,
        **kwargs)
    return dataset

def show_batch(dataset):
  for batch, label in dataset.take(2):
    for key, value in batch.items():
        # sentence = value.numpy()
        # sentence = np.array([x.decode() for x in sentence])#convert from array string
        # sentence = str("{}".format(sentence))#convert string to utf8
        # sentence = sentence[2:-2]#remove [' characteters from string
        # print(sentence)
        # docs.append(sentence)
        print("{:20s}: {}".format(key,value))


def pack(features, labels):
    return tf.stack(list(features.values()), axis = -1), labels

csv_columns = ['storyid', 'storytitle', 'key 1', 'key 2', 'key 3', 'key 4', 'key 5']

storyplot_dataset = get_dataset(train_file_path, select_columns=csv_columns)

show_batch(storyplot_dataset)


# t = Tokenizer()
# t.fit_on_texts(docs)
# print(len(t.word_index)+1)

#29219 words for storyplot csv
#38617 for full data

#TODO build tokenizer for data
#TODO get vocab size

#TODO split dataset into training and testing / evaluation
#TODO build plot planning model train and test

#just for storyline planning
#seq2seq conditional generation model
#encodes title into a vector using a bidirectional long short-term memory
#network
#generates words in the storyline using another single-directional LSTM

