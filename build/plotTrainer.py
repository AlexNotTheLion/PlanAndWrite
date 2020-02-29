from __future__ import absolute_import, division, print_function, unicode_literals

import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

csv_file_path = "storyplot.csv"
vocab_file = "vocab.txt"

getVocab = False
storeData = False

vocab = []

with open(csv_file_path) as f:
    csvLen = sum(1 for row in f)

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

# def show_batch(dataset):
    # for batch, label in dataset.take(2):
    #     for key, value in batch.items():
    #     # sentence = value.numpy()
    #     # sentence = np.array([x.decode() for x in sentence])#convert from array string
    #     # sentence = str("{}".format(sentence))#convert string to utf8
    #     # sentence = sentence[2:-2]#remove [' characteters from string
    #     # print(sentence)
    #     # docs.append(sentence)
    #         print("{:20s}: {}".format(key,value))


def pack(features, labels):
    return tf.stack(list(features.values()), axis = 1), labels


csv_columns = ['storyid', 'storytitle', 'key1', 'key2', 'key3', 'key4', 'key5']

storyplot_dataset = get_dataset(csv_file_path, select_columns=csv_columns)

packed_dataset = storyplot_dataset.map(pack)


print("loading vocab from csv")
df = pd.read_csv(csv_file_path, usecols=[1,2,3,4,5,6], encoding="utf-8")
vocab = df[['storytitle', 'key1', 'key2', 'key3', 'key4', 'key5']].agg(' '.join, axis=1).tolist()
print("file loaded")

t = Tokenizer()
t.fit_on_texts(vocab)
print("token size : {}".format(len(t.word_index)))

# b = Tokenizer()
# b.fit_on_texts(vocab2)
# print("token size b : {}".format(len(b.word_index)))

# test_seq = t.texts_to_sequences(test_data)
# print(test_seq)

# for item in packed_dataset.take(1):
#     print(item)

# print("Please enter a title")
# title = input()
# print("you entered: {}".format(title))

# model = keras.Sequential(

# )

#shape is the shape of the data array e.g. a picture has size x, size y, color depth, for my data its 1,6 (1 story, 6 sentences inc title)
#from that get a basic prediction working, add a title and get a crappy probably bad sentence

#paper:
#using a seq to seq model, conditional genration model, first encoded title into a vecotr using a BILSTM, 
#and generates words in storyline using a single direction LSTM

#vocab sizes
#29219 words for storyplot csv
#38617 for full data

#TODO 1. encode title into vector

#TODO split dataset into training and testing / evaluation
#TODO build plot planning model train and test

#just for storyline planning
#seq2seq conditional generation model
#encodes title into a vector using a bidirectional long short-term memory
#network
#generates words in the storyline using another single-directional LSTM

