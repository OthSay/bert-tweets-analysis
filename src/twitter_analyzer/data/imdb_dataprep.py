#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 21:19:57 2020

@author: sayemothmane
"""

# importing IMDB dataset

import os 
import pandas as pd 
from tqdm import tqdm
import pickle

data_path = "/Users/sayemothmane/ws/research/nlp/projectx/data/aclImdb"

train_path = os.path.join(data_path, "train")
test_path = os.path.join(data_path, "test")
subdirs = ["unsup", "pos", "neg"]

# train data
x_train = []
y_train= []
for subdir in subdirs : 
    path = os.path.join(train_path, subdir)    
    for file in tqdm(os.listdir(path)) : 

        filepath = os.path.join(path, file)
        with open(filepath, "r") as f : 
            text = f.readlines()
            
        x_train.append(text[0])
        y_train.append(subdir)

        
# test data
x_test= []
y_test=[]

for subdir in subdirs : 
    path = os.path.join(test_path, subdir)
    if os.path.isdir(path):
        for file in tqdm(os.listdir(path)) : 
            
            filepath = os.path.join(path, file)
            with open(filepath, "r") as f : 
                text = f.readlines()
            
            x_test.append(text[0])
            y_test.append(subdir)


train_data = pd.DataFrame({"text" : x_train, "category" : y_train})
test_data = pd.DataFrame({"text" : x_test, "category" : y_test})


# Saving data to pickle
with open(os.path.join(data_path, "pickle", "train.pkl"), "wb") as f:
    pickle.dump(train_data, f)

with open(os.path.join(data_path, "pickle", "test.pkl"), "wb") as f:
    pickle.dump(test_data, f)
    
    
