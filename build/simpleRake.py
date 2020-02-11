from __future__ import absolute_import, division, print_function, unicode_literals

import rake
import functools
import operator
import numpy as np

import tensorflow as tf

r = rake.Rake("SmartStoplist.txt")

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
    for batch in dataset.take(15):
        story = ""
        for key, value in batch.items():
            sentence = value.numpy()
            sentence = np.array([x.decode() for x in sentence])#convert from array string
            sentence = str("{}".format(sentence))#convert string to utf8
            sentence = sentence[2:-2]#remove [' characteters from string
            out = str("{:12s}: {}".format(key, sentence))
            print(out)

            if (key == "storyid" or key == "storytitle"):
                continue

            story += sentence
        key = r.run(story)#extract most important word from given text
        keysents = [k for k, s in key.items()]
        sorted_keysents = sorted(keysents, key=operator.itemgetter(1,2))
        for k, si, wi in sorted_keysents:#k is the key word, si is sentence index, and wi is word index
            print("k: {}, si: {}, wi: {} \n".format(k,si,wi))
        print("----------------------------------------------------------")

csv_columns = ['storyid', 'storytitle', 'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']

temp_dataset = get_dataset(train_file_path, select_columns=csv_columns)

show_batch(temp_dataset)
