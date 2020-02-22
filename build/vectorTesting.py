from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import functools
import numpy as np
import tensorflow_datasets as tfds
import os

from keras_preprocessing.text import Tokenizer

file_names = ['cowper.txt','derby.txt','butler.txt']

def labeler(example, index):
    return example, tf.cast(index, tf.int64)

labeled_data_sets = []
docs = []

for i, file_name in enumerate(file_names):
    lines_datset = tf.data.TextLineDataset(file_name)
    labeled_dataset = lines_datset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_dataset)

buffer_size = 50000
batch_size = 64
take_size = 5000

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

# for ex in all_labeled_data.take(5):
#     print(ex)

tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
for text_tensor, _ in all_labeled_data.take(3):
    sentence = text_tensor.numpy()
    # sentence = np.array([x.decode() for x in sentence])#convert from array string
    sentence = str("{}".format(sentence))#convert string to utf8
    sentence = sentence[2:-1]#remove [' characteters from string
    # print(sentence)
    docs.append(sentence)
    some_tokens = tokenizer.tokenize(text_tensor.numpy())
    # print(some_tokens) 
    vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)
print("tfds token size : {}".format(vocab_size))

t = Tokenizer()
t.fit_on_texts(docs)
print("keras token size : {}".format(len(t.word_index)))

# print(t.word_index)

# print("keras token size : {}".format(len(t.word_index)+1))
encoder = tfds.features.text.TokenTextEncoder(t.word_index)
example_text = next(iter(all_labeled_data))[0].numpy()
print(example_text)

encoded_example = encoder.encode(example_text)
print(encoded_example)