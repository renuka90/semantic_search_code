

import re
import nltk
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

# Lemmatize with POS Tag

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# 1. Init Lemmatizer
lmtzr = WordNetLemmatizer()


# Read line by line, run lemmatization (is that still the right way to go, though?) and write result line by line to new file
datafile = "C:/Thesis/Data/save/Master_Data/pre_processed_data/data_preprocessed_txt.txt"
tgtfile  = "C:/Thesis/Data/save/Master_Data/lemmatized_data/data_lemmatized.txt"

cutoff_word_length = 2
cutoff_sent_length = 5

with open(tgtfile, 'a', encoding='utf8') as outfile:
    for line in open(datafile, 'r', encoding='utf8'):
        temp = re.sub('\n', '', line)
        words = temp.split(' ')
        lemmas = [lmtzr.lemmatize(w, get_wordnet_pos(w)) for w in words if len(w) > cutoff_word_length]
        if (len(lemmas) > cutoff_sent_length):
            lemmas = ' '.join(lemmas) + '\n'
            outfile.write(lemmas)






