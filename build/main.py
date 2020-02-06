from __future__ import absolute_import, division, print_function, unicode_literals

from rake_nltk import Rake, Metric

import functools

import numpy as np

import tensorflow as tf

r = Rake(max_length = 2)

train_file_path = "ROCStories.csv"


def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
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
            out = str("{:12s}: {}".format(key, sentence))
            print(out)

            if (key == "storyid" or key == "storytitle"):
                continue

            r.extract_keywords_from_text(sentence)#extract most important word from given text
            print("best ranked word: {}".format(r.get_ranked_phrases_with_scores()))
            print("\n")
        print("\n")


csv_columns = ['storyid', 'storytitle', 'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']

temp_dataset = get_dataset(train_file_path, select_columns=csv_columns)

show_batch(temp_dataset)
