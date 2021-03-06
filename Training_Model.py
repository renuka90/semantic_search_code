#!/usr/bin/env python
# coding: utf-8

# # Train all text Data Set

# In[1]:


#  line-based iterator that reads the file one line at a time instead of reading everything in memory at once
import os
class MySentences(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
        for line in open(self.filename,encoding='utf-8'):
            yield line.split()


# In[2]:

# sentences is kept as a memory-friendly iterator and the contents of the txt file are NEVER fully loaded into memory
sentences = MySentences('C:/Thesis/Data/save/Master_Data/lemmatized_data/data_lemmatized_latest.txt') 

def getWordFreq(corpus):
    result = {}
    for data in corpus:
        for word in data:
            if word in result:
                result[word] += 1 #adding result in the dictionary
            else:
                result[word] = 1

    return result #returning full dict


# In[ ]:


fdist1 = getWordFreq(sentences)


# In[ ]:


# check whether given key already exists in a dictionary. 
def checkKey(dict, key): 
      
    if key in dict.keys(): 
        print("Present, ", end =" ") 
        print("value =", dict[key]) 
    else: 
        print("Not present") 


df_list = df['word_list'].values.astype(str)

# convert dict to df
import pandas as pd
df_fdist = pd.DataFrame.from_dict(fdist1, orient='index')

df_fdist = df_fdist.sort_values(by=[0], ascending=False)

# set the threshold to remove the certain section of vocabulary
theta = 0.96    
df_threshold = df_fdist[df_fdist[0].cumsum()/df_fdist[0].sum() < theta]

minValue = df_threshold[0].min()
print(minValue)
print(len(df_threshold))

df_threshold


#save dataframe to excel
df_threshold.to_excel("C:/Thesis/Data/save/Master_Data/Model/latest/word2vec/word_occurance_list.xlsx")  

# some memory clean-up
del fdist1
del df_fdist
del df_threshold


# # train the model

from gensim.models import Word2Vec
epochs=200
#sentences = list_sent
model_bestpara = Word2Vec(
        sentences, # our dataset
        size=100, # this is the length of the vector to numerically represent the "meaning" of words
        window=15, # this is the number of neighboring words to consider when assigning "meaning" to a word
        min_count=minValue, # minimum number of occurrences
        alpha = 0.005,
        iter =  epochs) # this is how fast the model adapts its representation of the "meaning" of a word

print(model_bestpara)

import pandas as pd
pd.DataFrame(model_bestpara.wv.most_similar(positive = ['engagement'], topn=10), columns = ['word', 'similarity'])

# load the model1
from gensim.models import Word2Vec

model_bestpara = Word2Vec.load("C:/Thesis/Data/save/Master_Data/Model/latest/word2vec/word2vec_model1_96_percentile.model")

print(model_bestpara)

# load the model2
from gensim.models import Word2Vec

model_2 = Word2Vec.load("C:/Thesis/Data/save/Master_Data/Model/latest/word2vec/word2vec_model2_96_percentile.model")


import pandas as pd
pd.DataFrame(model_2.wv.most_similar(positive = ['engagement'], topn=10), columns = ['word', 'similarity'])


# load the model3
from gensim.models import Word2Vec

model_3 = Word2Vec.load("C:/Thesis/Data/save/Master_Data/Model/latest/word2vec/word2vec_model3_96_percentile.model")


import pandas as pd
pd.DataFrame(model_3.wv.most_similar(positive = ['engagement'], topn=10), columns = ['word', 'similarity'])


import pandas as pd
pd.DataFrame(model_bestpara.wv.most_similar(positive = ['satisfaction'], topn=10), columns = ['word', 'similarity'])


# save the model
model_bestpara.save('C:/Thesis/Data/save/Master_Data/Model/latest/word2vec/word2vec_model1.model') 

