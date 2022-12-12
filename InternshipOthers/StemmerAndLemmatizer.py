from nltk.stem import PorterStemmer

stemming = PorterStemmer()
words = ["programmer", "programming", "studies", "corpora", "goes"]

print("Printing the words with stemming:")
for w in words:
    print(w, " : ", stemming.stem(w))

'''Output:
programmer  :  programm
programming  :  program
studies  :  studi
corpora  :  corpora
goes  :  goe
'''

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
# words = ["programmer", "programming", "studies", "corpora", "goes"]

print("Printing the words with lemmatization:")
for w in words:
    print(w, " : ", lemmatizer.lemmatize(w))

'''Output:
programmer  :  programmer
programming  :  programming
studies  :  study
corpora  :  corpus
goes  :  go
'''