from __future__ import absolute_import, division, print_function, unicode_literals

import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from keras_preprocessing.text import Tokenizer

train_file_path = "storyplot.csv"

vocab = []
vocab_new = []

with open(train_file_path) as f:
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


csv_columns = ['storyid', 'storytitle', 'key 1', 'key 2', 'key 3', 'key 4', 'key 5']

storyplot_dataset = get_dataset(train_file_path, select_columns=csv_columns)

packed_dataset = storyplot_dataset.map(pack)

for item, _ in packed_dataset:
    for text in item:
        for t in text:
            sentence = t.numpy()
            sentence = str("{}".format(sentence))#convert string to utf8
            sentence = sentence[2:-1]#remove [' characteters from string
            vocab.append(sentence)
           
t = Tokenizer()
t.fit_on_texts(vocab)
encoder = tfds.features.text.TokenTextEncoder(t.word_index)
print("token size : {}".format(encoder.vocab_size))

# for item, _ in packed_dataset.take(1):
#     for text in item:
#         for t in text:
#             sentence = t.numpy()
#             sentence = str("{}".format(sentence))#convert string to utf8
#             sentence = sentence[2:-1]#remove [' characteters from string
#             print(sentence)
#             encoded_t = encoder.encode(sentence)
#             print(encoded_t)

def encode(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label

def encode_map_fn(text, label):
    encoded_text, label = tf.py_function(encode,inp=[text, label], Tout=(tf.int64, tf.int64))

    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text, label

all_encoded_data = packed_dataset.map(encode_map_fn)

print(all_encoded_data.dtype)


# BUFFER_SIZE = 50000
# BATCH_SIZE = 64
# TAKE_SIZE = 5000

# train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
# train_data = train_data.padded_batch(BATCH_SIZE)

# test_data = all_encoded_data.skip(TAKE_SIZE)
# test_data = test_data.padded_batch(BATCH_SIZE)

# train_batches = train_data.shuffle(1000).padded_batch(10)
# test_batch = test_data.shuffle(1000).padded_batch(10)

# train_batch, train_labels = next(iter(train_batches))
# print(train_batch.numpy())

model = keras.Sequential([
    keras.layers.Embedding(encoder.vocab_size, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])

# model.complie(optimizer='adam', 
# loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
# metrics=['accuracy'])

# history = model.fit(train_batch, epochs = 10, validation_data=test_batch, validation_steps = 20)

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

