import re
import numpy as np
import pandas as pd

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
    '''Load the training data from a file in the HF format ( label \t text)
    takes as input the amount of lines to read from the begining. Left empty to read all the lines.  '''
    
    print(f"Going to read all data")
    tweets, labels = read_file_HF("./HF_data.txt", 0, amount)
    print(f"Loaded {len(tweets)} tweets!")
    return tweets, labels

def load_train_data_per_batch(amount_per_batch, iteration):
    '''Used to load the training data in batches instead of all at the same times, for low memory systems.
    Not used in this code since 32go of ram is enough to run everything without issues.'''

    starting_line = iteration * amount_per_batch
    end_line = starting_line + amount_per_batch
    print(f"Going to read {amount_per_batch*2} lines ({amount_per_batch} in each of the pos and neg datasets), starting_line:{starting_line}, end_line:{end_line}")
    tweets, labels = read_file_HF("./HF_data.txt", starting_line, end_line)
    print(f"Loaded {len(tweets)} tweets!")


    return tweets, labels

def read_file_HF(file_name, starting_line=0, end_line=0):
    '''Helper function to read a file with the HF format ( label \t text).
    takes the filename, the starting line and the ending line as parameters.
    Will return the requested lines in 2 numpy arrays teets, labels.'''
    tweets, labels = [], []
    with open( file_name, 'r', encoding='utf-8') as f:
        tweets = [line.split("\t",maxsplit=1) for line in f.readlines()[starting_line:end_line]]
        tweets = np.array(tweets)
        labels = tweets[:,0].astype(np.uint8)
        tweets = tweets[:,1]
        
    return(tweets, labels)

def check_libs():
    '''Helper function to see if cuda is correctly installed, Unused in this code.'''
    print(torch.version.cuda)
    print(torch.cuda.is_available() )
    # cupy.show_config()
    nltk.download('punkt')
    return(torch.cuda.is_available())

def load_tweets(full=False):
    '''Old method of reading the files, in case the HF dataset.txt hasn't been generated.
    This is deprecated, and load_train_data() should be used instead.'''
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
    '''Preprocessing function to apply on the raw tweet's text. 
    takes 3 params as entry: labels: np.array, tweets: np.array, token_tool: int
    token_tool serves as index to which method of tokenization to use on the tweets.

    returns:  X_train,X_test,y_train, y_test , shuffled and with test size being 0.005
    Currently the best performing is the TFID_VECT from sklearn.

    token_tool=0: uses the tokenizer from the bertweet model from hugging face. Took this one since
    it was one of the best performing one. How ever it yields poor results on the baselines because 
    it produces sparse results which don't perform well on simple models.

    token_tool=1: Uses the Count vectorizer from sklearn, This has been disabled as it did not produce good results,
    and took quite a while to run.

    token_tool=2: Uses sklearn hashvectorizer, it yields pretty good results and is also quite fast.

    token_tool=3: This is the one performing the best results, it uses sklearn TFID vectorizer. 

    token_tool=4: This one is disabled as running with glove resulted in issues with hardware limations/space.

    '''
    
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

    if should_split:
        X_train, X_test, y_train, y_test = train_test_split(tweets_out,labels,test_size = 0.005, random_state = 12345)

    return X_train,X_test,y_train, y_test

def plot_roc_auc(y_test,y_pred,name="Default"):
    '''Code taken online to produce simple plots, to help quickly compare the models/tokenizers.
    takes the true y and the predicted y, and a name for the model'''
    
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
    '''Generates the word2vec format from the glove huge dataset.
    only required if working with token_tool=4 '''
    word2vec_output_file = "./glove.twitter.27B.200d.word2vec"
    glove2word2vec("./glove.twitter.27B.200d.txt", word2vec_output_file)

def main():

    # dl_wordlists() # enable to download wordlists if needed
    # check_libs() # 
    # gen_word2vec_file() # only needed for the  TOKENIZER_GENSIM_WORD2VEC
    tweets, labels = load_train_data(500000)
    preprocessing_methods = ["TOKENIZER_HF_TOKENIZER_128" ,"TOKENIZER_SKLEARN_CountVectorizer" ,"TOKENIZER_SKLEARN_HASHVECTORIZER" ,"TOKENIZER_SKLEARN_TFID_VECT" ,"TOKENIZER_GENSIM_WORD2VEC" ]
    should_skip_method = [False,True,False,False,True] # Used to quickly enable/disable tokenization methods for testing which one produces best results.
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

        XGB_CLASSIFIER = xgb.XGBClassifier()
    

        model_fit(BNBmodel,X_train,X_test,y_train, y_test,name=f"BernoulliNB-{method_name}")
        model_fit(SVCmodel,X_train,X_test,y_train, y_test,name=f"LinearSVC-{method_name}")
        model_fit(LRmodel,X_train,X_test,y_train, y_test,name=f"LogisticRegression-{method_name}") 
        
        model_fit(XGB_CLASSIFIER,X_train,X_test,y_train, y_test,name=f"XGB_CLASSIFIER-{method_name}")
        model_fit(RandomForestClassifiermodel,X_train,X_test,y_train, y_test,name=f"RandomForestClassifier-{method_name}")
        
        # model_fit(NUSVCmodel,X_train,X_test,y_train, y_test,name="NonLinearSVC-{method_name}") # takes for ever to run..
    



def model_fit(model,X_train,X_test,y_train, y_test, name="Default"):
    '''Simple helper function to fit a given model (must follow sklearn api)
    and evaluate the results.'''
    print("Starting to fit model: "+name)
    model.fit(X_train, y_train)

    y_pred = model_Evaluate(model,X_test, y_test,name=name)
    # y_pred1 = model.predict(X_test)
    plot_roc_auc(y_test,y_pred,name=name)
    

def model_Evaluate(model,X_test, y_test,name="Default"):
    '''Function taken online to provide simple plots, and prints 
    the classification report from sklearn.
    takes as input the model, the x_test, y_test, and a name for the model plots/prints/save path'''
    
    # prediction and results
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Plots , using online code.
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1} n {v2}' for v1, v2 in zip(group_names,group_percentages)]
    print(labels)
    labels = np.asarray(labels).reshape(2,2)
    plt.figure()
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
    xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title (f"Confusion Matrix model:{name}", fontdict = {'size':18}, pad = 20)
    plt.savefig(f"./images/Confusion_Matrix_model_{name}.png")
    return y_pred



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device:",device)
    batch_size = 32

    main()

