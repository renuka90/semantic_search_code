# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 00:51:45 2020

@author: Renu
"""
import streamlit as st
from gensim.models import Word2Vec
import re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer 
from textblob import TextBlob 
import math

nltk.download('punkt')
nltk.download('wordnet')

#Jaro-winkler distance function
def jaro_similarity(s1, s2):
    """
    Computes the Jaro similarity between 2 sequences from:
    The Jaro distance between is the min no. of single-character transpositions
    required to change one word into another. The Jaro similarity formula from
        jaro_sim = 0 if m = 0 else 1/3 * (m/|s_1| + m/s_2 + (m-t)/m)
    where:
        - |s_i| is the length of string s_i
        - m is the no. of matching characters
        - t is the half no. of possible transpositions.
    """   
    # First, store the length of the strings
    # because they will be re-used several times.
    len_s1, len_s2 = len(s1), len(s2)

    # The upper bound of the distanc for being a matched character.
    match_bound = math.floor( max(len(s1), len(s2)) / 2 ) - 1

    # Initialize the counts for matches and transpositions.
    matches = 0  # no.of matched characters in s1 and s2
    transpositions = 0  # no. transpositions between s1 and s2

    # Iterate through sequences, check for matches and compute transpositions.
    for ch1 in s1:     # Iterate through each character.
        if ch1 in s2:  # Check whether the
            pos1 = s1.index(ch1)
            pos2 = s2.index(ch1)
            if(abs(pos1-pos2) <= match_bound):
                matches += 1
                if(pos1 != pos2):
                    transpositions += 1

    if matches == 0:
        return 0
    else:
        return 1/3 * ( matches/len_s1 +
                       matches/len_s2 +
                      (matches-transpositions//2) / matches
                     )
#***********************************************************


st.title('Semantic Search Engine test repo')
st.markdown('<style>h1{color: #bc0031;}</style>', unsafe_allow_html=True)
st.subheader('Find the similar terms.')
# load model data
model = Word2Vec.load('./data/model/word2vec_model3.model')

# get model vocab
get_model = model.wv.vocab
#define list to store model
list_model = []
for i in get_model:
    list_model.append(i)

# function to get the closest word with highest similarity score
def calcJaroDistance(word, numWords):

    temp = (word.split())

    if len(temp) > 1:
              
        word_list =[]
        word_len =[]
        
        for i in temp:
            
            dictWordDist = []
            word_sim = []
            
            for line in list_model:  
                
                wordDistance = jaro_similarity(i, line)
                word_sim.append(float(wordDistance))
                dictWordDist.append(line)

            d = {'word':dictWordDist,'word_sim':word_sim}

            # Convert list to pandas
            df = pd.DataFrame(d)
            df = df.sort_values(by='word_sim', ascending=False)
            df = df.loc[(df['word_sim']<=1)]
            #print(df.iloc[:numWords])
            word_list.append(df['word'].iloc[0])
            word_len.append(df.iloc[:numWords])
        return word_list
    else: 
     
        dictWordDist = []
        word_sim = []

        for line in list_model:          
            wordDistance = jaro_similarity(word, line)
            word_sim.append(float(wordDistance))
            dictWordDist.append(line)

        d = {'word':dictWordDist,'word_sim':word_sim}

        # Convert list to pandas
        df = pd.DataFrame(d)
        df = df.sort_values(by='word_sim', ascending=False)

        return df.iloc[:numWords]

#get input word
pos_str = st.text_input('Enter keyword(s)')
pos_str = pos_str.lower()

# convert the model into vectors
word_vectors = model.wv
    
# Tokenize: Split the sentence into words
word_list = nltk.word_tokenize(pos_str)

lemmatizer = WordNetLemmatizer()
# Lemmatize list of words and join
pos_str = ' '.join([lemmatizer.lemmatize(w) for w in word_list])

#textblob for spelling correction
check_spel = pos_str
pos_str = TextBlob(check_spel)  
 
try:
    if check_spel in word_vectors.vocab:
        
        pos_str = str(pos_str)
        # remove spaces both in the beginning and in the end of of string
        pos_str = re.sub("^\s+|\s+$", "", pos_str, flags=re.UNICODE)
        
        # any input that is NOT a-z, A-Z, 0-9,-,*
        pos_str = re.sub('[^a-zA-Z0-9-_*.]', ' ', pos_str)
        pos_str = re.sub(' +',' ',  re.sub('\W', ' ', pos_str))
        
        pos_words= pos_str.split(' ')
        
        
        if (len(pos_words[0]) > 0):
        
            st.write('SIMILAR TO ', pos_str)
            df = pd.DataFrame(model.wv.most_similar(positive = pos_words, topn=10), columns = ['SIMILAR_word', 'similarity'])
            df1 = df[['SIMILAR_word']]
            link_list = []
            for i in df['SIMILAR_word']:
                          
                word = 'https://scholar.google.nl/scholar?hl=nl&as_sdt=0%2C5&q=' + i
         
                link_list.append(word)
             # rename column as SIMILAR for UI
            
            df1['SIMILAR'] = link_list
                          
            df1 = df1[['SIMILAR']]
            def make_clickable(link):
                 # target _blank to open new window
                 # extract clickable text to display for your link
                 text = link.split('=')[3]
                 return f'<a target="_blank" href="{link}">{text}</a>'
             
             # link is the column with hyperlinks
            df1['SIMILAR'] = df1['SIMILAR'].apply(make_clickable)
            df1 = df1.rename(index={0:'1',1:'2',2:'3',3:'4',4:'5',5:'6',6:'7',7:'8',8:'9',9:'10'})
            df1 = df1.to_html(escape=False)
            st.write(df1, unsafe_allow_html=True)
                
    else:
        if check_spel != str(pos_str.correct()): 
            
            temp = calcJaroDistance(check_spel, 1)
            df_data = pd.DataFrame(temp)
            df_data = df_data.rename(columns={0: "word"})
            temp2 = df_data['word']
            pos_str = temp2.to_string(index=False)
            
            # prints the closest similary term
            st.markdown('<p style="color:red"> Did you mean: {}</p>'.format(str(pos_str)), unsafe_allow_html=True)
            st.markdown('<p style="color:Blue"> Showing result for : {}</p>'.format(str(pos_str)), unsafe_allow_html=True)
    
        pos_str = str(pos_str)
        
         # remove spaces both in the beginning and in the end of of string
        pos_str = re.sub("^\s+|\s+$", "", pos_str, flags=re.UNICODE)
        
        # any input that is NOT a-z, A-Z, 0-9,-,*
        pos_str = re.sub('[^a-zA-Z0-9-_*.]', ' ', pos_str)
        pos_str = re.sub(' +',' ',  re.sub('\W', ' ', pos_str))
        
        pos_words= pos_str.split(' ')
        
        
        if (len(pos_words[0]) > 0):
        
            st.write('SIMILAR TO ', pos_str)
            df = pd.DataFrame(model.wv.most_similar(positive = pos_words, topn=10), columns = ['SIMILAR_word', 'similarity'])
            df1 = df[['SIMILAR_word']]
            link_list = []
            for i in df['SIMILAR_word']:
                          
                word = 'https://scholar.google.nl/scholar?hl=nl&as_sdt=0%2C5&q=' + i
         
                link_list.append(word)
             # rename column as SIMILAR for UI
            
            df1['SIMILAR'] = link_list
             
             #st.dataframe(df1)
             
            df1 = df1[['SIMILAR']]
            def make_clickable(link):
                 # target _blank to open new window
                 # extract clickable text to display for your link
                 text = link.split('=')[3]
                 return f'<a target="_blank" href="{link}">{text}</a>'
             
             # link is the column with hyperlinks
            df1['SIMILAR'] = df1['SIMILAR'].apply(make_clickable)
            df1 = df1.rename(index={0:'1',1:'2',2:'3',3:'4',4:'5',5:'6',6:'7',7:'8',8:'9',9:'10'})
            df1 = df1.to_html(escape=False)
            st.write(df1, unsafe_allow_html=True)
except KeyError:
    st.markdown('<h4 style="color:#bc0031;"> WE ARE SORRY, THE SEARCH TERM IS NOT AVAILABLE IN OUR DATABASE.</h4>', unsafe_allow_html=True) 