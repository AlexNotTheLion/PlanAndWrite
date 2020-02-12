from __future__ import absolute_import, division, print_function, unicode_literals

import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras

train_file_path = "storyplot.csv"


def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(file_path,
        batch_size=1,
        label_name = None,
        na_value="",
        num_epochs=1,
        ignore_errors=True,
        shuffle=False,
        **kwargs)
    return dataset

def show_batch(dataset):
    for batch in dataset.take(1):
        for key, value in batch.items():
            sentence = value.numpy()
            sentence = np.array([x.decode() for x in sentence])#convert from array string
            sentence = str("{}".format(sentence))#convert string to utf8
            sentence = sentence[2:-2]#remove [' characteters from string
            print(str("{:12s}: {} \n".format(key, sentence)))


csv_columns = ['storytitle', 'key 1', 'key 2', 'key 3', 'key 4', 'key 5']

storyplot_dataset = get_dataset(train_file_path, select_columns=csv_columns)

#show_batch(temp_dataset)


#just for storyline planning
#seq2seq conditional generation model
#encodes title into a vector using a bidirectional long short-term memory
#network
#generates words in the storyline using another single-directional LSTM

(train_data, test_data), info = tfds.load(
    storyplot_dataset, 
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True, as_supervised=True)

encoder = info.features[storyplot_dataset].encoder
encoder.subwords[:20]

mnist = tf.keras.datasets.mnist

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
    ])