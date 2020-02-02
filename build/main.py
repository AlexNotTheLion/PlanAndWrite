from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np

import tensorflow as tf

train_file_path = "ROCStories.csv"

np.set_printoptions(precision = 3, suppress=True)

label_column = 'storyid'

def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=1,
        label_name = label_column,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
        **kwargs)
    return dataset

#raw_data = get_dataset(train_file_path)

def show_batch(dataset):
    for batch, label in dataset.take(5):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))
        print("\n")

#show_batch(raw_data)

csv_columns = ['storyid', 'storytitle', 'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']
#csv_columns = ['storyid', 'storytitle', 'sentence1']

temp_dataset = get_dataset(train_file_path, select_columns=csv_columns)

show_batch(temp_dataset)