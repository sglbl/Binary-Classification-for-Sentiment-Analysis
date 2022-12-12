import numpy as np
import pandas as pd
import tensorflow as tf
import urllib

from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense, SpatialDropout1D, Dropout, Convolution1D
from tensorflow.python.keras.layers import Flatten, LSTM, GlobalMaxPooling1D
from tensorflow.python.keras.layers import Embedding
# from tensorflow.python.keras import TextVectorization
from tensorflow.keras.layers import TextVectorization

def loadingFile(url):
    stms = []
    file = urllib.request.urlopen(url)

    for line in file:
        line = line.decode("utf-8")
        if( len(line) > 2 ):
            stm = line.strip()
            stms.append(stm)
    return stms

# wishUrl = 'wish.txt'; curseUrl = 'curse.txt'
wishUrl = 'https://raw.githubusercontent.com/kmkarakaya/ML_tutorials/master/data/dua.txt'
curseUrl= 'https://raw.githubusercontent.com/kmkarakaya/ML_tutorials/master/data/beddua.txt'
wish = loadingFile(wishUrl)
curse = loadingFile(curseUrl)

totalWish = len(wish)
totalCurse = len(curse)
print("Total wish: ", totalWish, " Total Curse: " , totalCurse)

# Wish size is 177 and curse size is 801 so making curse size 177 as well
curse = curse[:totalWish] # now there will be 177 curse
totalCurse = len(curse)
print("Now totalCurse is ", totalCurse)

# Making 2 parts as test and train data
# 10% is test data and other is training data

testWish= int(totalWish* 0.1)
testCurse = int(totalCurse * 0.1)
print('Total test data for Wish ', testWish)
print('Total test data for Curse ', testCurse)

# Rest of them will be train data
trainDocs = wish[:-testWish] + curse[:-testCurse]
testDocs  = wish[-testWish:] + curse[-testCurse:]
print("Length of train and test docs")
print(len(trainDocs)) 
print(len(testDocs)) 

trainLabels = np.concatenate((np.ones(totalWish-testWish),np.zeros(totalCurse-testCurse)), axis=0) 
testLabels = np.concatenate((np.ones(testWish),np.zeros(testCurse)), axis=0) 
print("Length of train and test labels")
print(len(trainLabels)) 
print(len(testLabels))

print("All docs ")
allDocs = trainDocs + testDocs
print("Length: ", len(allDocs), " and All Docs:")
print(allDocs)

#Tokenizing the corpus
# from tensorflow.python.keras.preprocessing import Tokenizer

# Tokenize our training data
tokenizer = TextVectorization()
tokenizer.fit_on_texts(allDocs)

document_count = tokenizer.document_count
vocab_size = len(tokenizer.word_index)

# Encode training data sentences into sequences
allDocs_sequences = tokenizer.texts_to_sequences(allDocs)

# Get max training sequence length
max_length = max([len(x) for x in allDocs_sequences])

# Get our training data word index
word_index = tokenizer.word_index
print("Corpus Summary")
print("Word index:", word_index)
print("document count  :", document_count)
print("vocabulary size :", vocab_size)
print("Maximum length of the statements :", max_length)

