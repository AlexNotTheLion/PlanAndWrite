from __future__ import absolute_import, division, print_function, unicode_literals

import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import time

csv_file_path = "storyplot.csv"

input_vocab = []
output_vocab = []

def tokenize(lang):
    lang_tokienizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokienizer.fit_on_texts(lang)
    tensor = lang_tokienizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokienizer

def load_dataset(path, examples):
    #load data and build sepparate tokenizers for input and targets
    print("loading vocab from csv")
    df = pd.read_csv(path, usecols=[1,2,3,4,5,6], encoding="utf-8", nrows=examples)
    Keys = df[['key1', 'key2', 'key3', 'key4', 'key5']].agg(' '.join, axis=1)#dataset of plotkeys
    Titles = df[['storytitle']]#dataset of titles
    #build a dataset of tiles matched with their plots
    sub_set = pd.concat(['<start> ' + Titles + ' <end>', '<start> ' + Keys + ' <end>'], axis = 1)# combined dataset of both titles and keys
    data = list(sub_set.itertuples(index=False, name =None))#convert to tuples list
    print("file loaded")

    features, labels = zip(*data)
    
    input_tensor, inp_lang_tokenizer = tokenize(features)
    target_tensor, targ_lang_tokenizer = tokenize(labels)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

def convert(lang, tensor):
    for t in tensor:
        if t!= 0:
            print ("%d ----> %s" % (t, lang.index_word[t]))


#loading plots data from csv
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(csv_file_path, None)


# convert(inp_lang, input_tensor[0])
# convert(targ_lang, target_tensor[0])

#TODO look into bi directional encoder-decoder models


##start comment

max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
# print( example_input_batch.shape, example_target_batch.shape)

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BhadanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BhadanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    
    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        attention_weights = tf.nn.softmax(score, axis = 1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis =1)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attention = BhadanauAttention(self.dec_units)
    
    def call(self, x, hidden, enc_ouput):
        context_vector, attention_weights = self.attention(hidden, enc_ouput)

        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis = -1)

        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, state, attention_weights

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

attention_layer = BhadanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)), sample_hidden, sample_output)


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss/int(targ.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

#load previously trained network
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
#continue
#run for 50  epochs, reset using single word storyplots
EPOCHS = 50

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for(batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} batch {} loss {:.4f}'.format(epoch+1,batch, batch_loss.numpy()))
    
    if(epoch + 1) % 2 ==0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} loss {:.4f}'.format(epoch+1, total_loss/steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    

    inputs = [inp_lang. word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen = max_length_inp, padding = 'post')

    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)
    
    return result, sentence, attention_plot


def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    # attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]


translate("bob went to the store")

#After 50 epochs reached 0.4019
#sunday june 21st 11am
#sample input "bob went to the store"
#output "store steak asked groceries wife store store"

# dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor))

# for i in dataset.take(1):
#     print(i)




