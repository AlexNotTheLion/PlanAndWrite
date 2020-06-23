from __future__ import absolute_import, division, print_function, unicode_literals

import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import time
import tensorflow_addons as tfa
import itertools

csv_file_path = "storyplot.csv"

input_vocab = []
output_vocab = []

query = True


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


def max_len(tensor):
    return max(len(t) for t in tensor)

#loading plots data from csv
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(csv_file_path, None)

x_train, x_test, y_train, y_test = train_test_split(input_tensor, target_tensor, test_size=0.2)
BATCH_SIZE = 64
BUFFER_SIZE = len(x_train)
steps_per_epoch = BUFFER_SIZE
embedding_dims = 256
rnn_units = 1024
dense_units = 1024
Dtype = tf.float32

Tx = max_len(input_tensor)
Ty = max_len(target_tensor)

input_vocab_size = len(inp_lang.word_index)+1
output_vocab_size = len(targ_lang.word_index)+1
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


#try replacing with a bidirectional lstm to encode
class EncoderNetwork(tf.keras.Model):
    def __init__(self, input_vocab_size, embedding_dims, rnn_units):
        super().__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size, output_dim=embedding_dims)

        self.encoder_rnnlayer = tf.keras.layers.LSTM(rnn_units,return_sequences=True, return_state=True)
        # self.encoder_rnnlayer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_units,return_sequences=True, return_state=True))


class DecoderNetwork(tf.keras.Model):
    def __init__(self, output_vocab_size, embedding_dims, rnn_units):
        super().__init__()
        self.decoder_embedding = tf.keras.layers.Embedding(input_dim=output_vocab_size, output_dim=embedding_dims)

        self.dense_layer = tf.keras.layers.Dense(output_vocab_size)
        self.decoder_rnncell = tf.keras.layers.LSTMCell(rnn_units)

        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        self.attention_mechanism = self.build_attention_mechanism(dense_units, None, BATCH_SIZE*[Tx])
        self.rnn_cell = self.build_rnn_cell(BATCH_SIZE)
        
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.dense_layer)

    def build_attention_mechanism(self, units, memory, memory_sequence_length):
        return tfa.seq2seq.BahdanauAttention(units, memory = memory, memory_sequence_length=memory_sequence_length)

    def build_rnn_cell(self, batch_size):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnncell, self.attention_mechanism, attention_layer_size=dense_units)
        return rnn_cell

    def build_decoder_initial_state(self, batch_size, encoder_state,Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size = batch_size, dtype = Dtype)

        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state

encoderNetwork = EncoderNetwork(input_vocab_size, embedding_dims, rnn_units)
decoderNetwork = DecoderNetwork(output_vocab_size, embedding_dims, rnn_units)
optimizer = tf.keras.optimizers.Adam()

def loss_function(y_pred, y):

    sparsecategoricalcrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction='none')
    loss = sparsecategoricalcrossentropy(y_true=y, y_pred=y_pred)
    mask = tf.logical_not(tf.math.equal(y,0))
    mask = tf.cast(mask, dtype = loss.dtype)
    loss = mask*loss
    loss = tf.reduce_mean(loss)
    return loss

def train_step(input_batch, output_batch, encoder_initial_cell_state):

    loss = 0
    with tf.GradientTape() as tape:
        encoder_emb_inp = encoderNetwork.encoder_embedding(input_batch)
        a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp, initial_state = encoder_initial_cell_state)
        # a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer.layer(encoder_emb_inp, initial_state = encoder_initial_cell_state)


        decoder_input = output_batch[:,:-1]
        decoder_output = output_batch[:,1:]

        decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)

        decoderNetwork.attention_mechanism.setup_memory(a)
        decoder_initial_state = decoderNetwork.build_decoder_initial_state(BATCH_SIZE, encoder_state=[a_tx, c_tx], Dtype=tf.float32)

        outputs, _, _ = decoderNetwork.decoder(decoder_emb_inp, initial_state=decoder_initial_state, sequence_length=BATCH_SIZE*[Ty-1])

        logits = outputs.rnn_output

        loss = loss_function(logits, decoder_output)

    variables = encoderNetwork.trainable_variables + decoderNetwork.trainable_variables
    gradients = tape.gradient(loss, variables)

    grads_and_vars = zip(gradients, variables)
    optimizer.apply_gradients(grads_and_vars)
    return loss

def initialize_initial_state():
    return [tf.zeros((BATCH_SIZE, rnn_units)), tf.zeros((BATCH_SIZE, rnn_units))]

epoch = 15

checkpoint_dir = './training_checkpoints_new'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoderNetwork, decoder=decoderNetwork)

#load checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
if query == False:
    for i in range(1, epoch+1):
        start = time.time()
        encoder_initial_cell_state = initialize_initial_state()
        total_loss = 0.0

        for(batch,(input_batch, output_batch)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(input_batch, output_batch, encoder_initial_cell_state)
            total_loss += batch_loss
            if(batch+1)%100 == 0:
                print("Total loss: {} Epoch: {} Batch: {}".format(batch_loss.numpy(), i , batch+1))

        if(i + 1) % 2 ==0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Epoch {} loss {:.4f}'.format(i, total_loss/steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

if query == True:

    input_raw = 'bob went to the store'

    input_lines= ['<start> ' + input_raw+'']
    input_sequences = [[inp_lang.word_index[w] for w in line.split(' ')] for line in input_lines]
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=Tx,padding='post')

    inp = tf.convert_to_tensor(input_sequences)
    inference_batch_size = input_sequences.shape[0]
    encoder_initial_cell_state = [tf.zeros((inference_batch_size, rnn_units)), tf.zeros((inference_batch_size, rnn_units))]

    encoder_emb_inp = encoderNetwork.encoder_embedding(inp)
    a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp, initial_state=encoder_initial_cell_state)

    start_tokens = tf.fill([inference_batch_size], targ_lang.word_index['<start>'])
    end_token = targ_lang.word_index['<end>']
    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

    decoder_input = tf.expand_dims([targ_lang.word_index['<start>']]* inference_batch_size, 1)
    decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)

    decoder_instance = tfa.seq2seq.BasicDecoder(cell = decoderNetwork.rnn_cell, sampler = greedy_sampler, output_layer=decoderNetwork.dense_layer)
    decoderNetwork.attention_mechanism.setup_memory(a)

    decoder_initial_state = decoderNetwork.build_decoder_initial_state(inference_batch_size, encoder_state=[a_tx, c_tx], Dtype = tf.float32)

    # max_iterations is the number of words to predict
    # maximum_iterations = tf.round(tf.reduce_max(Tx)*2)
    maximum_iterations = 5

    decoder_embedding_matrix = decoderNetwork.decoder_embedding.variables[0]
    (first_finished, first_inputs, first_state) = decoder_instance.initialize(decoder_embedding_matrix, start_tokens = start_tokens, end_token = end_token, initial_state=decoder_initial_state)

    inputs = first_inputs
    state = first_state
    predictions = np.empty((inference_batch_size, 0), dtype = np.int32)

    for j in range(maximum_iterations):
        outputs, next_state, next_inputs, finished = decoder_instance.step(j, inputs, state)
        inputs = next_inputs
        state = next_state
        outputs = np.expand_dims(outputs.sample_id, axis = -1)
        predictions = np.append(predictions, outputs, axis = -1)

    #TODO look into bi directional encoder-decoder models

    print ("prompt: {}\n".format(input_raw))

    print("predictions")

    for i in range(len(predictions)):
        line = predictions[i,:]
        seq = list(itertools.takewhile(lambda index: index !=2, line))
        print(" ".join([targ_lang.index_word[w] for  w in seq]))

#epoch record
#sunday 21st june - 15
#monday 22nd june - 75
#total 100 rounds

#test Results
#input = "bob went to the store"
#results = "dinner started store needed done"



#NOTES
#neural generation models
# static is title only
#encodes using a biLSTM
#decodes / generates each work using a normal LSTM
#using bahdanau attention mechanism

#Title = t
#ti = word in title
#s = story
#si = sentence in a story
#l = storyline
#li = word in a storyline

#<EOT> end of title
