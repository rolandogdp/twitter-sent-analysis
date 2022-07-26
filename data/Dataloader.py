import os
import sys
import torch

def read_file(file_name_label_tuple, starting_line=0, end_line=0):
    fname, label = file_name_label_tuple
    tweets, labels = [], []
    with open( fname, 'r', encoding='utf-8') as f:
        tweets = [line for line in f.readlines()[starting_line:end_line]]
        labels = [label] * (end_line-starting_line)

    return(tweets, labels)

def load_train_data(amount_per_batch, iteration):
 
    if use_full_dataset == True:
        X_train_neg_path = project_path + "train_neg_full.txt"
        X_train_pos_path = project_path + "train_pos_full.txt"
        
    else:
        X_train_neg_path = project_path + "train_neg.txt"
        X_train_pos_path = project_path + "train_pos.txt"

    amount_per_batch = amount_per_batch // 2
    
    starting_line = iteration * amount_per_batch
    end_line = starting_line + amount_per_batch
    print(f"Going to read {amount_per_batch*2} lines, starting_line:{starting_line}, end_line:{end_line}")
    tweets, labels = read_file((X_train_neg_path, 0), starting_line=starting_line, end_line=end_line)
    tweets_2, labels_2 = read_file((X_train_pos_path, 1), starting_line=starting_line, end_line=end_line)
    tweets += tweets_2
    tweets_2 = []
    del(tweets_2)
    labels += labels_2
    labels_2 = []
    del(labels_2)
    print(f"Loaded {len(tweets)} tweets!")
    
    return np.array(tweets), np.array(labels)

def load_test_data():
    filename = project_path + "test_data.txt"
    tweets = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.partition(",")[2]   # this allows to get the text content only for each line of the file (see that the file starts with "n," where n is the line numbner); it parts the string into three strings as: before the arg, the arg, and after the arg
            tweets.append(line.rstrip())

    return tweets


def vectorize_data(tweets,train_indices,val_indices):
    vectorizer = CountVectorizer(max_features=5000)   # 5000

    # Important: we call fit_transform on the training set, and only transform on the validation set
    X_train = vectorizer.fit_transform(tweets[train_indices])
    X_val = vectorizer.transform(tweets[val_indices])

    return X_train, X_val