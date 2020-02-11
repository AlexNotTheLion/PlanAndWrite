from __future__ import absolute_import, division, print_function, unicode_literals

import rake
import operator

from progress.bar import IncrementalBar

import functools
import csv
import numpy as np

import tensorflow as tf


r = rake.Rake("SmartStoplist.txt",1,2)

train_file_path = "ROCStories.csv"
plot_file = "storyplot.csv"

with open(train_file_path) as f:
    row_count = sum(1 for row in f)
bar = IncrementalBar('building dataset', max=row_count)

with open('storyplot.csv', 'w', newline='') as f:
        fieldname = ['storyid','storyitle', 'key 1', 'key 2', 'key 3', 'key 4', 'key 5']
        writeCsv = csv.DictWriter(f, fieldnames = fieldname)

        writeCsv.writeheader()

def addStoryPlotCsv(file_path, sentenceList, title, storyid):
    with open('storyplot.csv', 'a', newline='') as af:
        fieldname = ['storyid','storytitle', 'key 1', 'key 2', 'key 3', 'key 4', 'key 5']
        write = csv.DictWriter(af, fieldnames = fieldname)

        write.writerow({'storyid' : storyid,
                            'storytitle' : title, 
                            'key 1' : sentenceList[0],
                            'key 2' : sentenceList[1],
                            'key 3' : sentenceList[2],
                            'key 4' : sentenceList[3],
                            'key 5' : sentenceList[4]})

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
    for batch in dataset:
        story = ""
        name = ""
        id = ""
        sentenceList = []
        for key, value in batch.items():
            keyItem = ("",-1) #create an empty list of tuples 
            sentence = value.numpy()
            sentence = np.array([x.decode() for x in sentence])#convert from array string
            sentence = str("{}".format(sentence))#convert string to utf8
            sentence = sentence[2:-2]#remove [' characteters from string
            #out = str("{:12s}: {}".format(key, sentence))
            #print(out)

            if (key == "storyid" ):
                id = sentence
                continue

            if(key == "storytitle"):
                name = sentence
                continue;

            key = r.run(sentence, sep=u'[,.?!]')#extract most important word from given sentence

            keysents = [k for k, s in key.items()]
            sorted_keysents = sorted(keysents, key=operator.itemgetter(1,2))#convert to ordererd list
            for k, si, wi in sorted_keysents:#k is the key word, si is sentence index, and wi is word index
                #store tuplet in list of word and value, if value is larger than previous value replace it
                t = (k, wi)
                if(keyItem[1] < t[1]):
                    keyItem = t
                    
                    
            sentenceList.append(keyItem[0])
        
        addStoryPlotCsv(plot_file, sentenceList, name, id)
        #print(sentenceList)
        bar.next()
        #print("----------------------------------------------------------")
    bar.finish()

csv_columns = ['storyid', 'storytitle', 'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']

temp_dataset = get_dataset(train_file_path, select_columns=csv_columns)

show_batch(temp_dataset)