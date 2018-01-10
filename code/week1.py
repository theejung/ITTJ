import os, sys
import pandas as pd
import numpy as np
import zipfile

path = '../data/'
train_zipnm = 'train.tsv.zip'

#road train data into pandas DataFrame
with zipfile.ZipFile(os.path.abspath(path + train_zipnm)) as z:
    with z.open(z.namelist()[0]) as f:
        train = pd.read_csv(f, header = 0, delimiter = '\t')

#tokenize!
from nltk.tokenize import RegexpTokenizer

def token(array):
    out = list()
    tokenizer = RegexpTokenizer(r'\w+')
    for i in range(len(array)):
        out.append(tokenizer.tokenize(array[i].lower()))
    
    return out

train_sentences = token(train.Phrase)
print 'Number of Train', len(train_sentences)

#input should be 2d array. one array represents splitted words from dataset. How to tokenize it?
#Word2Vec -> Using gensim package at Python! (Of course we can also implement it with naive code and my train set!)
import gensim
embed_size = 300
#model = gensim.models.Word2Vec(train_sentences, size = embed_size, workers = 4)
#Load Pre-computed Model with GoogleNews
print 'Loading word2vec....'
model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(path, 'GoogleNews-vectors-negative300.bin.gz'), binary=True)



#print model.wv.most_similar(positive = ['apple'])[:5]
#print model.wv['apple']
#print model['apple']
#import pdb; pdb.set_trace()
#sys.exit(1)
#I will using this model's weight vector as an input for mine!
#I just assign '0' : if there is no words in Phrase or the words were not in our word2vec

def find_word2vec(array):
    out = list()
    for sentence in array:
        temp_vect = []
        num_word = 0
        for word in sentence:
            try:
                if temp_vect == []:
                    temp_vect = model[word]
                else:
                    temp_vect = temp_vect + model[word]
                
                num_word +=1
            except KeyError:
                continue
        
        if num_word == 0:
            out.append(np.zeros(embed_size))
        else:
            out.append(temp_vect/num_word)
    
    return out

#import pdb; pdb.set_trace()
X = find_word2vec(train_sentences)
Y = train.Sentiment

#Data split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, random_state = 63)
#print sum(Y_train), sum(Y_test)


##Data Normalization
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

#Modeling Fitting and Prediction
from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier

# Need to add: Hyperparameter(Batch Size / max_iter) Tuning with 10% of Validation set!)
#SVM
svm_model = svm.LinearSVC()
svm_model.fit(X_train, Y_train)
svm_pred = svm_model.predict(X_test)

#logit
logit_model = linear_model.LogisticRegression(C=1e5)
logit_model.fit(X_train, Y_train)
logit_pred = logit_model.predict(X_test)

#MLP
mlp_model = MLPClassifier(hidden_layer_sizes=(128,128), batch_size = 1024, max_iter = 500) #Solver default = 'adam'
mlp_model.fit(X_train, Y_train)
mlp_pred = mlp_model.predict(X_test)


#Simple accuracy
#SVM
cmp_svm = [int(x==y) for x,y in zip(svm_pred, Y_test)]
print(np.mean(cmp_svm))
#Logit
cmp_logit = [int(x==y) for x,y in zip(logit_pred, Y_test)]
print(np.mean(cmp_logit))
#MLP
cmp_mlp = [int(x==y) for x,y in zip(mlp_pred, Y_test)]
print(np.mean(cmp_mlp))

#Confusion Matrix
#from sklearn.metrics import confusion_matrix

def confusion_matrix(expected, predicted, n_classes):
    m = [[0] * n_classes for i in range(n_classes)]
    for pred, exp in zip(predicted, expected):
        m[pred][exp] += 1
    return m

svm_conf = confusion_matrix(Y_test, svm_pred, len(np.unique(Y)))
logit_conf = confusion_matrix(Y_test, logit_pred, len(np.unique(Y)))
mlp_conf = confusion_matrix(Y_test, mlp_pred, len(np.unique(Y)))
print mlp_conf #Col: label from predicted model, #Row: label from True value

#Precision & Recall: Should be calculated each class!
def prec_rec(array):
    precision_out = list()
    recall_out = list()
    fscore_out = list()
    for i in range(len(array)):
        recall_out.append(round(array[i][i]*1.0/sum(array[i])*100,2))
        precision_tmp = [x[i] for x in array]
        
        if sum(precision_tmp) > 0:
            precision_out.append(round(array[i][i]*1.0/sum(precision_tmp)*100,2))
        else:
            precision_out.append(0)

        precision_, recall_ = precision_out[-1], recall_out[-1]
        fscore_out.append(2*(precision_*recall_)/((precision_ + recall_)*100))
            
    return precision_out, recall_out, fscore_out

print prec_rec(svm_conf) 
print prec_rec(logit_conf)
print prec_rec(mlp_conf)

#T_SNE
#Load X!
df = pd.read_csv(os.path.abspath(path + 'Word2Vec_result'), header = 0)
X = df.drop(df.columns[[0, 301]], axis = 1)
Y = df['Sentiment']

from matplotlib import pyplot as plt
from tsne import bh_sne
import matplotlib

#Reshape to 'float64' to be used in tsne package
x_data = np.asarray(X).astype('float64')
x_data = x_data.reshape((x_data.shape[0], -1))

n = 10000
x_data = x_data[:n]
y_data = Y[:n]

#perform t-SNE embedding with 10,000 samples [Dim_reduction of word2vec]
vis_data = bh_sne(x_data)

# plot the result.. hmm...
## T-SNE : Circle small, 
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]
num_category = len(np.unique(y_data))
colors = ['#F21D0D', '#DF4959','#F4B8E4','#6E7DF7','#1A2ECB'] #0,1,2,3,4
plt.scatter(vis_x, vis_y, c=y_data, s= 2, cmap=matplotlib.colors.ListedColormap(colors))
cb = plt.colorbar()
loc = np.arange(0,max(y_data),1)
cb.set_ticks(loc)
plt.savefig()
plt.close()

## T-SNE with  Top 1K Freq words
from collections import Counter
train_sentence_combine = [item for sublist in train_sentences for item in sublist]
k1_freq = Counter(train_sentence_combine).keys()[:1000]

arr_word = list()
k1_freq_new = list()
for i in k1_freq:
    try:
        arr_word.append(model[i])
        k1_freq_new.append(i)
    except KeyError:
        continue

arr_word = np.asarray(arr_word).astype('float64')
arr_word = arr_word.reshape((arr_word.shape[0], -1))

vis_word = bh_sne(arr_word)
vis_xx, vis_yy = vis_word[:,0], vis_word[:,1]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(vis_xx, vis_yy, s= 0.5)

for i, txt in enumerate(k1_freq_new[:100]):
    ax.annotate(txt, (vis_xx[i], vis_yy[i]), size = 6)
    
plt.savefig()
plt.close()