import os
import pandas as pd
import zipfile

path = 'ITTJ/data/'
train_zipnm = 'train.tsv.zip'
test_zipnm = 'test.tsv.zip'

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

#input should be 2d array. one array represents splitted words from dataset. How to tokenize it?
#Word2Vec -> Using gensim package at Python! (Of course we can also implement it with naive code and my train set!)
import gensim
import numpy as np

embed_size = 300
#model = gensim.models.Word2Vec(train_sentences, size = embed_size, workers = 4)
#Load Pre-computed Model with GoogleNews
model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(path, 'GoogleNews-vectors-negative300.bin.gz'), binary=True)

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
	
X = find_word2vec(train_sentences)
Y = train.Sentiment

#Data split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1)

##Data Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Modeling Fitting and Prediction
from sklearn.neural_network import MLPClassifier

# Need to add: Hyperparameter(Batch Size / max_iter) Tuning with 10% of Validation set!)
mlp_model = MLPClassifier(hidden_layer_sizes=(128,128), batch_size = 1024, max_iter = 500) #Solver default = 'adam'
mlp_model.fit(X_train, Y_train)
Y_score = mlp_model.predict(X_test)
cmp = [int(x==y) for x,y in zip(Y_score, Y_test)]

#Simple accuracy
print(np.mean(cmp))
#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_score)
print confusion_matrix #Col: label from predicted model, #Row: label from True value
#Precision & Recall: Should be calculated each class!

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
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]
num_category = len(np.unique(y_data))
colors = ['yellow', 'green','blue','purple','red'] #0,1,2,3,4
plt.scatter(vis_x, vis_y, c=y_data, cmap=matplotlib.colors.ListedColormap(colors))
cb = plt.colorbar()
loc = np.arange(0,max(y_data),1)
cb.set_ticks(loc)
plt.show()
