from __future__ import absolute_import, division, print_function, unicode_literals

import rake
import operator
import colorama
from colorama import Fore,Style
colorama.init()


from progress.bar import IncrementalBar

import functools
import csv
import numpy as np

import tensorflow as tf


r = rake.Rake("SmartStoplist.txt",1,1)

ROC17 = "ROC17.csv"
ROC16 = "ROC16.csv"
plot_file = "storyplot_test.csv"

with open(ROC17) as f:
    rc17 = sum(1 for row in f)

with open(ROC16) as f:
    rc16 = sum(1 for row in f)
bar = IncrementalBar('building dataset', max=rc17 + rc16 - 2)

with open(plot_file, 'w', newline='') as f:
        fieldname = ['storyid','storytitle', 'key1', 'key2', 'key3', 'key4', 'key5']
        writeCsv = csv.DictWriter(f, fieldnames = fieldname)

        writeCsv.writeheader()

def addStoryPlotCsv(file_path, sentenceList, title, storyid):
    with open(file_path, 'a', newline='', encoding='utf-8') as af:
        fieldname = ['storyid','storytitle', 'key1', 'key2', 'key3', 'key4', 'key5']
        write = csv.DictWriter(af, fieldnames = fieldname)

        write.writerow({'storyid' : storyid,
                            'storytitle' : title, 
                            'key1' : sentenceList[0],
                            'key2' : sentenceList[1],
                            'key3' : sentenceList[2],
                            'key4' : sentenceList[3],
                            'key5' : sentenceList[4]})

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

            if (key == "storyid"):
                id = sentence
                continue

            if(key == "storytitle"):
                name = sentence
                continue
            
            # print(f'This is {Fore.GREEN}color{Style.RESET_ALL}!')
            key = r.run(sentence, sep=u'[,.?!]')#extract most important word from given sentence
            # print(f'\n{Fore.GREEN}Original sentence{Style.RESET_ALL}: ' + sentence)

            # print("\nf'{Fore.GREEN}Original sentence{Style.RESET_ALL}: {}".format(sentence))
            keysents = [k for k, s in key.items()]
            sorted_keysents = sorted(keysents, key=operator.itemgetter(1,2))#convert to ordererd list
            for k, si, wi in sorted_keysents:#k is the key word, si is sentence index, and wi is word index
                # print("key: {}   score: {}".format(k,wi))
                #store tuplet in list of word and value, if value is larger
                                                             #than previous value replace it
                t = (k, wi)
                if(keyItem[1] < t[1]):
                    keyItem = t
            
            # print(f'{Fore.BLUE}Highest rank key to be stored{Style.RESET_ALL}: ' + keyItem[0])

                    
            sentenceList.append(keyItem[0])
        # print("\n\n")
        
        addStoryPlotCsv(plot_file, sentenceList, name, id)
        #print(sentenceList)
        bar.next()
        #print("----------------------------------------------------------")
    bar.finish()

csv_columns = ['storyid', 'storytitle', 'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']

winter_dataset = get_dataset(ROC17, select_columns=csv_columns)
spring_dataset = get_dataset(ROC16, select_columns=csv_columns)

final_dataset = winter_dataset.concatenate(spring_dataset)

show_batch(final_dataset)
