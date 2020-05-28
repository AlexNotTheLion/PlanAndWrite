from __future__ import absolute_import, division, print_function, unicode_literals

import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

csv_file_path = "storyplot.csv"

vocab = []

def tokenize(lang):
    tensor = word_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor

def load_dataset(data):
    features, labels = zip(*data)
    
    input_tensor = tokenize(features)
    target_tensor = tokenize(labels)

    return input_tensor, target_tensor

def convert(lang, tensor):
    for t in tensor:
        if t!= 0:
            print ("%d ----> %s" % (t, word_tokenizer.index_word[t]))


print("loading vocab from csv")
df = pd.read_csv(csv_file_path, usecols=[1,2,3,4,5,6], encoding="utf-8", nrows=10)
vocab = df[['storytitle', 'key1', 'key2', 'key3', 'key4', 'key5']].agg(' '.join, axis=1).tolist()
sub_set = df[['storytitle','key1']]
data = list(sub_set.itertuples(index=False, name =None))#build list of tuples of story title followed by first sentence 
print("file loaded")

word_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
word_tokenizer.fit_on_texts(vocab)
#these need to be two sepparate vocabulary lists, one for input, one for output, allowing matching of one to the other during decoding

input_tensor, target_tensor = load_dataset(data)

print(input_tensor[0])

convert(word_tokenizer, input_tensor[0])
print("\noutput tensor")
convert(word_tokenizer, target_tensor[0])

dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor))

for i in dataset.take(1):
    print(i)

##organise into features and tensors before slcices https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices

###END


#meeting discussion
#build zipped dataset of input, expected output
#pass to embedding layer and follow https://www.tensorflow.org/tutorials/text/nmt_with_attention
#then try some seq 2 seq stuff

# print(type(storyplot_dataset))

# packed_dataset = storyplot_dataset.map(pack)
#maybe build dataset manually rather than get_dataset function

#model fit can require a td.data dataset which should return a tuple of (inputs, targets) , (titles, plots) 

# for features, labels in packed_dataset.take(1):
#     print(features)
#     print(labels)  

# print("loading vocab from csv")
# df = pd.read_csv(csv_file_path, usecols=[1,2,3,4,5,6], encoding="utf-8")
# vocab = df[['storytitle', 'key1', 'key2', 'key3', 'key4', 'key5']].agg(' '.join, axis=1).tolist()
# print("file loaded")


# t = Tokenizer()
# t.fit_on_texts(vocab)
# print("token size : {}".format(len(t.word_index)))

#data needs to be split into two np arrays, titles as inputs and sentences as outputs
#these need to be tokenized arrays
#history = model.fit(input, output, epochs, **kwargs)
#model.predict(["test title"]) then print the predicted stroyplot
# a dense layer is a layer which all ouputs go to all the different following nodes in the following layer

#features = inputs to the model (title)
#labels = outputs (generated storyline!)
#training example is a pair of the features and labels

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

