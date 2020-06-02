
from gensim.models import Word2Vec
import pandas as pd

# an iterator that reads the file one line at a time instead of reading everything in memory at once

class MySentences(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
        for line in open(self.filename,encoding='utf-8'):
            yield line.split()


sentences = MySentences('C:/Thesis/Data/save/Master_Data/lemmatized_data/data_lemmatized.txt') # a memory-friendly iterator


def getWordFreq(corpus):
    result = {}
    for data in corpus:
        for word in data:
            if word in result:
                result[word] += 1 #adding result in the dictionary
            else:
                result[word] = 1

    return result #returning full dict



fdist1 = getWordFreq(sentences)

# convert dict to df

df_fdist = pd.DataFrame.from_dict(fdist1, orient='index')



df_fdist = df_fdist.sort_values(by=[0], ascending=False)


# set the threshold to remove the certain section of vocabulary
theta = 0.95    
df_threshold = df_fdist[df_fdist[0].cumsum()/df_fdist[0].sum() < theta]



minValue = df_threshold[0].min()
print(minValue)
print(len(df_threshold))


# some memory clean-up
del fdist1
del df_fdist
del df_threshold

#train the model

epochs=200
model_bestpara = Word2Vec(
        sentences, # our dataset
        size=100, # this is the length of the vector to numerically represent the "meaning" of words
        window=15, # this is the number of neighboring words to consider when assigning "meaning" to a word
        min_count=minValue, # minimum number of occurrences
        alpha = 0.005,
        iter =  epochs) # this is how fast the model adapts its representation of the "meaning" of a word

#print(model_bestpara)


pd.DataFrame(model_bestpara.wv.most_similar(positive = ['performance'], topn=10), columns = ['word', 'similarity'])

# save the word2vec model 
#model_bestpara.save('C:/Thesis/Data/save/Master_Data/Model/latest/word2vec/word2vec_model1.model') 


