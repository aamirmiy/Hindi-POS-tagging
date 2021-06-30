# -*- coding: utf-8 -*-
"""
Created on Thu May 20 22:09:15 2021

@author: aamir
"""
# Importing required libraries
import streamlit as st
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import warnings
warnings.filterwarnings('ignore')
import pickle
import tensorflow as tf

from conllu import parse_incr
from io import open
from decimal import *
import collections
from nltk import word_tokenize

#Importing the necessary text files and pickle files which contain the saved models
lstm=tf.keras.models.load_model('lstm.h5')
infile = open('crfhindi_1.sav','rb')
model = pickle.load(infile)


with open("test.txt", "rb") as fp:
    Xtrain = pickle.load(fp)
with open("testy.txt", "rb") as fp1:
    ytrain = pickle.load(fp1)

 #Creating a widget for taking user's input
user_input = st.text_input("Enter hindi sentence", "")
models=['CRF','HMM','ANN'] #These are the three models that will be used for tagging
make_choice=st.selectbox('Select model:', models) #User can select anyone using a drop down menu

if(make_choice=='CRF'): #Depending on the model chosen , functions associated with that model will be executed
    @st.cache(suppress_st_warning=True)
    def extract_features(sentence, index): #This function is used to extract a set of features from a sentence
        return {
            'word': sentence[index],
            'is_first': index == 0,
            'is_last': index == len(sentence) - 1,
            'prefix-1': sentence[index][0],
            'prefix-2': sentence[index][:2],
            'prefix-3': sentence[index][:3],
            'prefix-3': sentence[index][:4],
            'suffix-1': sentence[index][-1],
            'suffix-2': sentence[index][-2:],
            'suffix-3': sentence[index][-3:],
            'suffix-3': sentence[index][-4:],
            'next_word': sentence[index + 1] if index < len(sentence) - 1 else '',
            'prev_word': '' if index == 0 else sentence[index - 1],
            'has_hyphen': '-' in sentence[index],
            'is_numeric': sentence[index].isdigit()
        }
    if st.button('Predict'): # If predict button is pressed then tagging will be performed on the user sentence
        list1 = []
        list1.append(word_tokenize(user_input))
        xtesting = []
        for index in range(len(list1)):
            arrange = []
            for i in range(len(list1[index])):
                arrange.append(extract_features(list1[index], i))
            xtesting.append(arrange)
        pred = model.predict(xtesting)
        st.write(str(pred[0])) #This the equivalent print function.

if(make_choice=='HMM'): #This is the Hidden Markov Model.
    tag_list = set()
    tag_count = {}
    word_set = set()

    @st.cache(suppress_st_warning=True)
    def transition_count(X, y):
        global tag_list
        global word_set
        transition_dict = {}
        global tag_count
        for v in range(len(X)):
            previous = "start"
            for data in range(len(X[v])):
                i = X[v][data]
                word = i
                word_set.add(word.lower())
                tag = y[v][data]
                tag_list.add(tag)
                if tag in tag_count:
                    tag_count[tag] += 1
                else:
                    tag_count[tag] = 1
                if (previous + "~tag~" + tag) in transition_dict:
                    transition_dict[previous + "~tag~" + tag] += 1
                    previous = tag
                else:
                    transition_dict[previous + "~tag~" + tag] = 1
                    previous = tag
        return transition_dict, tag_count, tag_list, word_set


    transmission_m, tag_count, tag_list, word_set = transition_count(Xtrain, ytrain)


    @st.cache(suppress_st_warning=True)
    def transition_probability(X, y):
        count_dict = transmission_m
        prob_dict = {}
        for key in count_dict:
            den = 0
            val = key.split("~tag~")[0]
            # Probabilty of a tagA to be followed by tagB out of all possible tags #
            for key_2 in count_dict:
                if key_2.split("~tag~")[0] == val:
                    den += count_dict[key_2]
            prob_dict[key] = Decimal(count_dict[key]) / (den)
        return prob_dict


    @st.cache(suppress_st_warning=True)
    def transition_smoothing(X, y):
        transition_prob = transition_probability(X, y)
        for tag in tag_list:
            # if a tag does not occur as a start tag, then set its probability to be a start tag to minimum value #
            if "start" + tag not in transition_prob:
                transition_prob[("start" + "~tag~" + tag)] = Decimal(1) / Decimal(len(word_set) + tag_count[tag])
        for tag1 in tag_list:
            for tag2 in tag_list:
                # if a particular tag combination does not exist in the dictionary, we set its probability to minimum#
                if (tag1 + "~tag~" + tag2) not in transition_prob:
                    transition_prob[(tag1 + "~tag~" + tag2)] = Decimal(1) / Decimal(len(word_set) + tag_count[tag1])
        return transition_prob


    @st.cache(suppress_st_warning=True)
    def emission_count(X, y):
        count_word = {}
        for v in range(len(X)):
            for data in range(len(X[v])):
                i = X[v][data]
                word = i
                tag = y[v][data]
                # map the words in the training set to their tagged POS #
                if word.lower() + "/" + tag in count_word:
                    count_word[word.lower() + "/" + tag] += 1
                else:
                    count_word[word.lower() + "/" + tag] = 1
        return count_word


    @st.cache(suppress_st_warning=True)
    def emission_probability(X, y):
        global tag_count
        word_count = emission_count(X, y)
        emission_prob_dict = {}
        # calculate probability of a word to be a certain Tag out of all the possible tags that it can be #
        for key in word_count:
            emission_prob_dict[key] = Decimal(word_count[key]) / tag_count[key.split("/")[-1]]
        return emission_prob_dict


    transition_model = transition_smoothing(Xtrain, ytrain)
    emission_model = emission_probability(Xtrain, ytrain)


    @st.cache(suppress_st_warning=True)
    def viterbi_algorithm(sentence, tag_list, transition_prob, emission_prob, tag_count, word_set):
        global tag_set
        # Get words from each sentence #
        sentence = sentence.strip("\n")
        word_list = sentence.split(" ")
        current_prob = {}
        for tag in tag_list:
            # transition probability #
            tp = Decimal(0)
            # Emission probability #
            em = Decimal(0)
            # Storing the probability of every tag to be starting tag #
            if "start~tag~" + tag in transition_prob:
                tp = Decimal(transition_prob["start~tag~" + tag])
            # Check for word in training data. If present, check the probability of the first word to be of given tag#
            if word_list[0].lower() in word_set:
                if (word_list[0].lower() + "/" + tag) in emission_prob:
                    em = Decimal(emission_prob[word_list[0].lower() + "/" + tag])
                    # Storing probability of current combination of tp and em #
                    current_prob[tag] = tp * em
            # Check for word in training data. If absent then probability is just tp#
            else:
                em = Decimal(1) / (tag_count[tag] + len(word_set))
                current_prob[tag] = tp

        if len(word_list) == 1:
            # Return max path if only one word in sentence #
            max_path = max(current_prob, key=current_prob.get)
            return max_path
        else:
            # Tracking from second word to last word #
            for i in range(1, len(word_list)):
                previous_prob = current_prob
                current_prob = {}
                locals()['dict{}'.format(i)] = {}
                previous_tag = ""
                for tag in tag_list:
                    if word_list[i].lower() in word_set:
                        if word_list[i].lower() + "/" + tag in emission_prob:
                            em = Decimal(emission_prob[word_list[i].lower() + "/" + tag])
                            # Find the maximum probability using previous node's(tp*em)[i.e probability of reaching to the previous node] * tp * em (Bigram Model) #
                            max_prob, previous_state = max((Decimal(previous_prob[previous_tag]) * Decimal(
                                transition_prob[previous_tag + "~tag~" + tag]) * em, previous_tag) for previous_tag in
                                                           previous_prob)
                            current_prob[tag] = max_prob
                            locals()['dict{}'.format(i)][previous_state + "~" + tag] = max_prob
                            previous_tag = previous_state
                    else:
                        em = Decimal(1) / (tag_count[tag] + len(word_set))
                        max_prob, previous_state = max((Decimal(previous_prob[previous_tag]) * Decimal(
                            transition_prob[previous_tag + "~tag~" + tag]) * em, previous_tag) for previous_tag in
                                                       previous_prob)
                        current_prob[tag] = max_prob
                        locals()['dict{}'.format(i)][previous_state + "~" + tag] = max_prob
                        previous_tag = previous_state

                # if last word of sentence, then return path dicts of all words #
                if i == len(word_list) - 1:
                    max_path = ""
                    last_tag = max(current_prob, key=current_prob.get)
                    max_path = max_path + last_tag + " " + previous_tag
                    for j in range(len(word_list) - 1, 0, -1):
                        for key in locals()['dict{}'.format(j)]:
                            data = key.split("~")
                            if data[-1] == previous_tag:
                                max_path = max_path + " " + data[0]
                                previous_tag = data[0]
                                break
                    result = max_path.split()
                    result.reverse()
                    return " ".join(result)

    if st.button('Predict'):
        path = viterbi_algorithm(user_input, tag_list, transition_model, emission_model, tag_count, word_set)
        word = user_input.split(" ")
        #word = word_tokenize(user_input)
        tag = path.split(" ")
        for j in range(0, len(word)):
            if (j == len(word) - 1):
                st.write(word[j] + "/" + tag[j] + u'\n')
            else:
                st.write(word[j] + "/" + tag[j] + " ")


if(make_choice=='ANN'): #This is the artificial Neural Network Model
    with open('fasttext.pkl', 'rb') as f:
        embeddings_index = pickle.load(f) #loading word embeddings
    word = embeddings_index.keys()
    word = list(word)
    word2id = {k: word.index(k) for k in word}
    EMBEDDING_DIM=300
    embedding_matrix = np.zeros((len(word2id)+1,EMBEDDING_DIM))
    for word,i in word2id.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector

    @st.cache(suppress_st_warning=True)
    def add_new_word(new_word,new_vector,new_index,embedding_matrix,word2id): #Function for handling words that are not present in the word embeddings.If such words are encoundtered then they are assigned the unknown token.
      embedding_matrix = np.insert(embedding_matrix, [new_index],[new_vector],axis=0)
      word2id = {word:(index+1) if index>=new_index else index for word,index in word2id.items()}
      word2id[new_word] = new_index
      return embedding_matrix,word2id


    UNK_index = 0
    UNK_token = "UNK"
    unk_vector = embedding_matrix.mean(0)
    embedding_matrix,word2id = add_new_word(UNK_token,unk_vector,UNK_index,embedding_matrix,word2id)

    @st.cache(suppress_st_warning=True)
    def flatten(y):
        l = []
        for i in y:
            for j in i:
                l.append(j)
        return l

    eos_index = 1
    eos_tag = "EOS"
    eos_vector = np.random.standard_normal(300)
    embedding_matrix, word2id = add_new_word(eos_tag, eos_vector, eos_index, embedding_matrix, word2id)

    yflat = flatten(ytrain)

    tag2id = {}
    for item in yflat:
        tag = item
        tag2id.setdefault(tag, len(tag2id))

    id2tag = {v: k for k, v in tag2id.items()}

    if st.button('Predict'):
        sentence = word_tokenize(user_input)
        test_set = []
        for ele in sentence:
            try:
                test_set.append(word2id[ele])
            except KeyError:
                test_set.append(word2id["UNK"])

        test_set = np.array(test_set)
        for i in range(len(test_set)):
            if (test_set[i] == 0):
                continue
            else:
                test_set[i] = test_set[i] - 1
        pred = lstm.predict(test_set)
        num = pred.shape[0]
        l = []
        for i in range(num):
            l.append((sentence[i], id2tag[np.argmax(pred[i])]))
        st.write(str(l))
