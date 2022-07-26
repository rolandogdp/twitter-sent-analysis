import os
import sys

import torch
from transformers import AutoTokenizer

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('<user>') else t
        t = 'http' if t.startswith('<url>') else t
        new_text.append(t)
    return " ".join(new_text)

def preprocess_texts(texts):
	new_texts = []
	for text in texts:
		new_texts.append(preprocess(text))
	return new_texts

def preprocess_bert(X, config):
	X_list = X.tolist()
	X_list = preprocess_texts(X_list)
	tokenizer = AutoTokenizer.from_pretrained(config.model_name)
	return tokenizer(X_list, truncation = True)


