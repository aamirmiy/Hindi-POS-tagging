# -*- coding: utf-8 -*-
"""
Created on Thu May 20 22:09:15 2021

@author: aamir
"""
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


#infile = open('C:/Users/aamir/Downloads/CDAC-POS-tagging/crfhindi_1.sav','rb')
infile = open('crfhindi_1.sav','rb')
model = pickle.load(infile)


file=open('hi_hdtb-ud-train.conllu','r',encoding='utf-8')
ud_files=[]
for tokenlist in parse_incr(file):
    ud_files.append(tokenlist)

lstm = tf.keras.models.load_model('hindi_nn_pos.h5')
#lstmfile = open('C:/Users/aamir/Downloads/lstmhindi.p','rb')
#lstm=pickle.load(lstmfile)

with open('fasttext.pkl','rb') as f:
  embeddings_index=pickle.load(f)

word=embeddings_index.keys()
word=list(word)
word2id={k:word.index(k) for k in word}

def dataset(ud_files):
    bank=[]
    for sentence in ud_files:
        tokens=[]
        tags=[]

        for token in sentence:
            tokens.append(token['form'])
            tags.append(token['upostag'])

        bank.append((tokens,tags))
    return bank

train=dataset(ud_files)



def separate(bank):
    X,y=[],[]
    for index in range(len(bank)):
        X.append(bank[index][0])
        y.append(bank[index][1])
    return X,y
    
Xtrain,ytrain=separate(train)

tag_list = set()
tag_count = {}
word_set = set()


def transition_count(X,y):
    global tag_list
    global word_set
    transition_dict = {}
    global tag_count
    for v in range(len(X)):
        previous="start"
        for data in range(len(X[v])):
            i=X[v][data]
            word = i
            word_set.add(word.lower())
            tag = y[v][data]
            tag_list.add(tag)

            if tag in tag_count:
                tag_count[tag]+=1
            else:
                tag_count[tag] = 1


            if (previous + "~tag~" + tag) in transition_dict:
                    transition_dict[previous + "~tag~" + tag] += 1
                    previous = tag
            else:
                    transition_dict[previous + "~tag~" + tag] = 1
                    previous = tag

    return transition_dict,tag_count,tag_list,word_set    

transmission_m,tag_count,tag_list,word_set = transition_count(Xtrain,ytrain) 



def transition_probability(X,y):
    #count_dict = transition_count(X,y)
    count_dict = transmission_m
    prob_dict = {}
    for key in count_dict:
        den = 0
        val = key.split("~tag~")[0]
        # Probabilty of a tagA to be followed by tagB out of all possible tags # 
        for key_2 in count_dict:
            if key_2.split("~tag~")[0] == val:
                den += count_dict[key_2]
        prob_dict[key] = Decimal(count_dict[key])/(den)
    return prob_dict



def transition_smoothing(X,y):
    transition_prob = transition_probability(X,y)
    for tag in tag_list:
    	# if a tag does not occur as a start tag, then set its probability to be a start tag to minimum value #
        if "start" + tag not in  transition_prob:
            transition_prob[("start" + "~tag~" + tag)] = Decimal(1) / Decimal(len(word_set) + tag_count[tag])
    for tag1 in tag_list:
        for tag2 in tag_list:
        	# if a particular tag combination does not exist in the dictionary, we set its probability to minimum#
            if (tag1 +"~tag~" + tag2) not in transition_prob:
                transition_prob[(tag1+"~tag~"+tag2)] = Decimal(1)/Decimal(len(word_set) + tag_count[tag1])
    return transition_prob



def emission_count(X,y):  
    count_word = {}
    for v in range(len(X)):
        for data in range(len(X[v])):
    #for value in train_data:
        #for data in value:
            i = X[v][data]
            word = i
            tag = y[v][data]
            # map the words in the training set to their tagged POS #
            if word.lower() + "/" + tag in count_word:
                count_word[word.lower() + "/" + tag] +=1
            else:
                count_word[word.lower() + "/" + tag] = 1
    return count_word



def emission_probability(X,y):
    global tag_count
    word_count = emission_count(X,y)
    emission_prob_dict = {}
    # calculate probability of a word to be a certain Tag out of all the possible tags that it can be #
    for key in word_count:
        emission_prob_dict[key] = Decimal(word_count[key])/tag_count[key.split("/")[-1]]
    return emission_prob_dict

transition_model = transition_smoothing(Xtrain,ytrain)
emission_model = emission_probability(Xtrain,ytrain)



def viterbi_algorithm(sentence, tag_list, transition_prob, emission_prob,tag_count, word_set):
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
        if "start~tag~"+tag in transition_prob:
            tp = Decimal(transition_prob["start~tag~"+tag])
        # Check for word in training data. If present, check the probability of the first word to be of given tag#
        if word_list[0].lower() in word_set:
            if (word_list[0].lower()+"/"+tag) in emission_prob:
                em = Decimal(emission_prob[word_list[0].lower()+"/"+tag])
                # Storing probability of current combination of tp and em #
                current_prob[tag] = tp * em
         # Check for word in training data. If absent then probability is just tp# 
        else:
            em = Decimal(1) /(tag_count[tag] +len(word_set))
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
                    if word_list[i].lower()+"/"+tag in emission_prob:
                        em = Decimal(emission_prob[word_list[i].lower()+"/"+tag])
                        # Find the maximum probability using previous node's(tp*em)[i.e probability of reaching to the previous node] * tp * em (Bigram Model) #
                        max_prob, previous_state = max((Decimal(previous_prob[previous_tag]) * Decimal(transition_prob[previous_tag + "~tag~" + tag]) * em, previous_tag) for previous_tag in previous_prob)
                        current_prob[tag] = max_prob
                        locals()['dict{}'.format(i)][previous_state + "~" + tag] = max_prob
                        previous_tag = previous_state
                else:
                    em = Decimal(1) /(tag_count[tag] +len(word_set))
                    max_prob, previous_state = max((Decimal(previous_prob[previous_tag]) * Decimal(transition_prob[previous_tag+"~tag~"+tag]) * em, previous_tag) for previous_tag in previous_prob)
                    current_prob[tag] = max_prob
                    locals()['dict{}'.format(i)][previous_state + "~" + tag] = max_prob
                    previous_tag = previous_state

            # if last word of sentence, then return path dicts of all words #
            if i == len(word_list)-1:
                max_path = ""
                last_tag = max(current_prob, key=current_prob.get)
                max_path = max_path + last_tag + " " + previous_tag
                for j in range(len(word_list)-1,0,-1):
                    for key in locals()['dict{}'.format(j)]:
                        data = key.split("~")
                        if data[-1] == previous_tag:
                            max_path = max_path + " " +data[0]
                            previous_tag = data[0]
                            break
                result = max_path.split()
                result.reverse()
                return " ".join(result)


def merge(list1, list2):    
    merged_list = []
    for i in range(max((len(list1), len(list2)))):
  
        while True:
            try:
                tup = (list1[i], list2[i])
            except IndexError:
                if len(list1) > len(list2):
                    list2.append('')
                    tup = (list1[i], list2[i])
                elif len(list1) < len(list2):
                    list1.append('')
                    tup = (list1[i], list2[i])
                continue
            merged_list.append(tup)
            break
    return merged_list


def extract_features(sentence, index):
    return{
      'word':sentence[index],
      'is_first':index==0,
      'is_last':index ==len(sentence)-1,
      'prefix-1':sentence[index][0],
      'prefix-2':sentence[index][:2],
      'prefix-3':sentence[index][:3],
      'prefix-3':sentence[index][:4],
      'suffix-1':sentence[index][-1],
      'suffix-2':sentence[index][-2:],
      'suffix-3':sentence[index][-3:],
      'suffix-3':sentence[index][-4:],
      'next_word':sentence[index+1] if index<len(sentence)-1 else '',
      'prev_word':'' if index == 0 else sentence[index-1],
      'has_hyphen': '-' in sentence[index],
      'is_numeric': sentence[index].isdigit()
    }

def flatten(y):
  l=[]
  for i in y:
    for j in i:
      l.append(j)
  return l

yflat=flatten(ytrain)

tag2id={}
for item in yflat:
  tag=item
  tag2id.setdefault(tag,len(tag2id))
  
id2tag={v:k for k,v in tag2id.items()}

eos_index = 1
eos_tag="EOS"

UNK_index = 0
UNK_token = "UNK"

context_size = 2

def get_data(tagged_words, word2id):
    x= []
    unk_count = 0
    
    span = 2*context_size+1 # total 5 words are being considered
    buffer = collections.deque(maxlen=span)
    padding = [(eos_tag, None)] * context_size
    buffer += padding + tagged_words[:context_size]
    
    for item in (tagged_words[context_size:] + padding):
        buffer.append(item)
        window_ids = np.array([word2id.get(word) if (word in word2id) else UNK_index for (word,_) in buffer])
        x.append(window_ids)
        
        middle_word, middle_tag = buffer[context_size]
        
        
        if middle_word not in word2id:
            unk_count += 1
            
    
    return np.array(x)

user_input = st.text_input("Enter hindi sentence", "")

if st.button('CRF Model'):
   list1=[]
   list1.append(user_input.split())
   xtesting=[]
   for index in range(len(list1)):
      arrange=[]
      for i in range(len(list1[index])):
          arrange.append(extract_features(list1[index],i))
      xtesting.append(arrange) 
   pred = model.predict(xtesting)
   st.write(str(pred[0]))

if st.button('HMM Model'):
    path = viterbi_algorithm(user_input, tag_list, transition_model, emission_model,tag_count, word_set)
    word = user_input.split(" ")
    tag = path.split(" ")
    for j in range(0,len(word)):
        if(j==len(word)-1):
            st.write(word[j] + "/" + tag[j]+ u'\n')
        else:
            st.write(word[j] + "/" + tag[j] + " ")
    
if st.button('LSTM Model'):
    sentence=word_tokenize(user_input)
    x1=[]
    for ele in sentence:
      x1.append((ele,None))
    xsample=get_data(x1,word2id)
    prediction=lstm.predict(xsample)
    num=prediction.shape[0]
    l=[]
    for i in range(num):
      l.append((sentence[i],id2tag[np.argmax(prediction[i])]))
    st.write(l)