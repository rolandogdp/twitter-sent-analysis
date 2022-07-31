# import numpy as np
# import spacy
# import torch
# import cupy
# utilities
import re
# from turtle import title
import numpy as np
import pandas as pd
# plotting
# import seaborn as sns
# from wordcloud import WordCloud
import matplotlib.pyplot as plt
# nltk
from nltk.stem import WordNetLemmatizer
# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


import numpy as np
import pandas as pd


from sklearn.metrics import roc_curve, auc

import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import nltk
from nltk.tokenize import word_tokenize,RegexpTokenizer,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer
import gensim

import xgboost as xgb

# from wordcloud import WordCloud
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import HashingVectorizer
import matplotlib.pyplot as plt



from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec




#TOKENIZER CONSTANTS:
TOKENIZER_HF_TOKENIZER_128 = 0
TOKENIZER_SKLEARN_CountVectorizer = 1
TOKENIZER_SKLEARN_HASHVECTORIZER = 2
TOKENIZER_SKLEARN_TFID_VECT = 3
TOKENIZER_GENSIM_WORD2VEC = 4



class Word2VecVectorizer:
  def init(self, model):
    print("Loading in word vectors...")
    self.word_vectors = model
    print("Finished loading in word vectors")

  def fit(self, data):
    pass

  def transform(self, data):
    # determine the dimensionality of vectors
    v = self.word_vectors.get_vector('king')
    self.D = v.shape[0]

    X = np.zeros((len(data), self.D))
    n = 0
    emptycount = 0
    for sentence in data:
      tokens = sentence.split()
      vecs = []
      m = 0
      for word in tokens:
        try:
          # throws KeyError if word not found
          vec = self.word_vectors.get_vector(word)
          vecs.append(vec)
          m += 1
        except KeyError:
          pass
      if len(vecs) > 0:
        vecs = np.array(vecs)
        X[n] = vecs.mean(axis=0)
      else:
        emptycount += 1
      n += 1
    print("Numer of samples with no words found: %s / %s" % (emptycount, len(data)))
    return X


  def fit_transform(self, data):
    self.fit(data)
    return self.transform(data)


def load_train_data(amount=-1):
    
    print(f"Going to read all data")
    tweets, labels = read_file_HF("./HF_data.txt", 0, amount) #TODO: CHange this value
    print(f"Loaded {len(tweets)} tweets!")


    return tweets, labels

def load_train_data_per_batch(amount_per_batch, iteration):
 

    starting_line = iteration * amount_per_batch
    end_line = starting_line + amount_per_batch
    print(f"Going to read {amount_per_batch*2} lines ({amount_per_batch} in each of the pos and neg datasets), starting_line:{starting_line}, end_line:{end_line}")
    tweets, labels = read_file_HF("./HF_data.txt", starting_line, end_line)
    print(f"Loaded {len(tweets)} tweets!")


    return tweets, labels

def read_file_HF(file_name, starting_line=0, end_line=0):
    tweets, labels = [], []
    with open( file_name, 'r', encoding='utf-8') as f:
        tweets = [line.split("\t",maxsplit=1) for line in f.readlines()[starting_line:end_line]]
        tweets = np.array(tweets)
        labels = tweets[:,0].astype(np.uint8)
        tweets = tweets[:,1]
        
    return(tweets, labels)

def check_libs():
    print(torch.version.cuda)
    print(torch.cuda.is_available() )
    # cupy.show_config()
    nltk.download('punkt')
    return(torch.cuda.is_available())

def load_tweets(full=False):
    if full:
       file_pos = "train_pos_full.txt"
       file_neg = "train_neg_full.txt"
    else:
        file_pos = "train_pos.txt"
        file_neg = "train_neg.txt"
    with open(file_pos,"r") as positives:
        positives_list = positives.read().split("\n")
    with open(file_neg,"r") as negatives:
        negatives_list = negatives.read().split("\n")

    print(f"type(positives_list){type(positives_list)}")
    
    labels = [1]*len(positives_list) + [0]*len(negatives_list)
    tweets = positives_list + negatives_list
    return labels,tweets
    
def preprocess_data(labels,tweets,token_tool=0):
    
    # lm = nltk.WordNetLemmatizer()
    # stop_words=set(stopwords.words("english"))
    # print("Tweet before processing:",tweets[0])

    should_split = True

    if token_tool == 0 :
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=True)
        tweets_out = tokenizer(tweets.tolist(), max_length=128, padding="max_length")
        tweets_out = np.array(tweets_out["input_ids"])
    elif token_tool == TOKENIZER_SKLEARN_CountVectorizer:
        count_vect = CountVectorizer()
        tweets_out = count_vect.fit_transform(tweets)

    elif token_tool ==  TOKENIZER_SKLEARN_HASHVECTORIZER:
        hv = HashingVectorizer()
        tweets_out = hv.transform(tweets)
    elif token_tool == TOKENIZER_SKLEARN_TFID_VECT:

        vectoriser = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.75,
        min_df=3,
        strip_accents="unicode",
        sublinear_tf=True)
        tweets_out = vectoriser.fit_transform(tweets)
        #print('No. of feature_words: ', len(vectoriser.get_feature_names()))
    elif token_tool == TOKENIZER_GENSIM_WORD2VEC:

        #glove_input_file = glove_filename
        word2vec_output_file = "./glove.twitter.27B.200d.word2vec"


        model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

        # Set a word vectorizer
        vectorizer = Word2VecVectorizer(model)
        # Get the sentence embeddings for the train dataset
        X_train, X_test, y_train, y_test = train_test_split(tweets,labels,test_size = 0.005, random_state = 12345)
        should_split = False
        X_train = vectorizer.fit_transform(X_train)
        
        # Get the sentence embeddings for the test dataset
        X_test = vectorizer.transform(X_test)

                
    # print("Tweet processed:",tweets[0,:])
    

    # input()
    if should_split:
        X_train, X_test, y_train, y_test = train_test_split(tweets_out,labels,test_size = 0.005, random_state = 12345)

    return X_train,X_test,y_train, y_test

def plot_roc_auc(y_test,y_pred,name="Default"):
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC CURVE for {name}')
    plt.legend(loc="lower right")
    plt.savefig(f"./images/ROC_CURVE_model_{name}.png")
    # plt.show()

def dl_wordlists():
    import nltk
    nltk.download('stopwords')

def gen_word2vec_file():
    word2vec_output_file = "./glove.twitter.27B.200d.word2vec"
    glove2word2vec("./glove.twitter.27B.200d.txt", word2vec_output_file)

def main():
    # dl_wordlists()
    # check_libs()
    # gen_word2vec_file()
    tweets, labels = load_train_data(500000)
    preprocessing_methods = ["TOKENIZER_HF_TOKENIZER_128" ,"TOKENIZER_SKLEARN_CountVectorizer" ,"TOKENIZER_SKLEARN_HASHVECTORIZER" ,"TOKENIZER_SKLEARN_TFID_VECT" ,"TOKENIZER_GENSIM_WORD2VEC" ]
    should_skip_method = [False,True,False,False,True]
    for i,method_name in enumerate(preprocessing_methods):
        if should_skip_method[i]:
            continue
        print(f"Currently trying preprocessing method: {method_name}")
        X_train,X_test,y_train, y_test = preprocess_data(labels,tweets,i)
    # del(labels)
    # del(tweets)


        BNBmodel = BernoulliNB()
        SVCmodel = LinearSVC()
        LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
        NUSVCmodel = svm.NuSVC(gamma="auto")
        RandomForestClassifiermodel = RandomForestClassifier(n_estimators=200, max_depth=10)
        # BiLSTM_SentimentAnalysismodel = BiLSTM_SentimentAnalysis()

        XGB_CLASSIFIER = xgb.XGBClassifier()
    

        model_fit(BNBmodel,X_train,X_test,y_train, y_test,name=f"BernoulliNB-{method_name}")
        model_fit(SVCmodel,X_train,X_test,y_train, y_test,name=f"LinearSVC-{method_name}")
        model_fit(LRmodel,X_train,X_test,y_train, y_test,name=f"LogisticRegression-{method_name}") 
        
        model_fit(XGB_CLASSIFIER,X_train,X_test,y_train, y_test,name=f"XGB_CLASSIFIER-{method_name}")
        model_fit(RandomForestClassifiermodel,X_train,X_test,y_train, y_test,name=f"RandomForestClassifier-{method_name}")
        
        # model_fit(NUSVCmodel,X_train,X_test,y_train, y_test,name="NonLinearSVC-{method_name}")
        # model_fit(BiLSTM_SentimentAnalysismodel,X_train,X_test,y_train, y_test,name="BiLSTM_SentimentAnalysismodel-{method_name}")
    



def model_fit(model,X_train,X_test,y_train, y_test, name="Default"):
    print("Starting to fit model: "+name)
    model.fit(X_train, y_train)
    model_Evaluate(model,X_train,X_test,y_train, y_test,name=name)
    y_pred1 = model.predict(X_test)
    plot_roc_auc(y_test,y_pred1,name=name)
    

def model_Evaluate(model,X_train,X_test,y_train, y_test,name="Default"):
    
    # Predict values for Test dataset
    y_pred = model.predict(X_test)
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1} n {v2}' for v1, v2 in zip(group_names,group_percentages)]
    print(labels)
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
    xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title (f"Confusion Matrix model:{name}", fontdict = {'size':18}, pad = 20)
    plt.savefig(f"./images/Confusion_Matrix_model_{name}.png")
    #plt.show()


class BiLSTM_SentimentAnalysis(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout) :
        super().__init__()

        # The embedding layer takes the vocab size and the embeddings size as input
        # The embeddings size is up to you to decide, but common sizes are between 50 and 100.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # The LSTM layer takes in the the embedding size and the hidden vector size.
        # The hidden dimension is up to you to decide, but common values are 32, 64, 128
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # We use dropout before the final layer to improve with regularization
        self.dropout = nn.Dropout(dropout)

        # The fully-connected layer takes in the hidden dim of the LSTM and
        #  outputs a a 3x1 vector of the class scores.
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x, hidden):
        """
        The forward method takes in the input and the previous hidden state 
        """

        # The input is transformed to embeddings by passing it to the embedding layer
        embs = self.embedding(x)

        # The embedded inputs are fed to the LSTM alongside the previous hidden state
        out, hidden = self.lstm(embs, hidden)

        # Dropout is applied to the output and fed to the FC layer
        out = self.dropout(out)
        out = self.fc(out)

        # We extract the scores for the final hidden state since it is the one that matters.
        out = out[:, -1]
        return out, hidden
    
    def init_hidden(self):
        return (torch.zeros(1, batch_size, 32), torch.zeros(1, batch_size, 32))





    


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device:",device)
    batch_size = 32

    main()
