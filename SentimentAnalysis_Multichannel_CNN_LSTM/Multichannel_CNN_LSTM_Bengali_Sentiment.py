
# coding: utf-8

# In[1]:


import sklearn
import numpy as np
from glob import glob
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline 

import string
from os import listdir
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from pickle import dump
from string import punctuation


# In[2]:


categories = ['positive', 'negative']


# In[3]:


# save a dataset to file
def save_dataset(dataset, filename):
	dump(dataset, open(filename, 'wb'))
	print('Saved: %s' % filename)


# In[4]:


# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


# In[5]:


import csv

stop_words = 'stopwords_bn.txt'
text_data = []

with open(stop_words, 'r', encoding='utf-8') as temp_output_file:
    reader = csv.reader(temp_output_file, delimiter='\n')
    for row in reader:
        text_data.append(row)

stop_word_list = [x[0] for x in text_data]


# In[6]:


import string 

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
    
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
    
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
    
	# filter out stop words
	stop_words = stop_word_list
	tokens = [w for w in tokens if not w in stop_words]
    
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	tokens = ' '.join(tokens)
	return tokens


# In[7]:


def load_data_and_labels(positive_data_file, negative_data_file):
    
    # load the positive doc
    pos_doc = load_doc(positive_data_file)
    
    # clean doc
    positive_examples = clean_doc(pos_doc)
    
    # load the negative doc
    neg_doc = load_doc(negative_data_file)
    negative_examples = clean_doc(neg_doc)
    
    # Split by words
    x_text = positive_examples + negative_examples
    
    # Generate labels
    positive_labels = [[0] for _ in positive_examples]
    negative_labels = [[1] for _ in negative_examples]
    trainy = [0 for _ in positive_examples] + [1 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    
    return [x_text, trainy]


# In[8]:


print("Loading data...")

#Training dataset
positive_data_file = 'C:/Users/admin-karim/Downloads/tmp/train/bangla.pos'
negative_data_file = 'C:/Users/admin-karim/Downloads/tmp/train/bangla.neg'

trainX, trainY = load_data_and_labels(positive_data_file, negative_data_file)

#Testing dataset
positive_data_file = 'C:/Users/admin-karim/Downloads/tmp/test/bangla.pos'
negative_data_file = 'C:/Users/admin-karim/Downloads/tmp/test/bangla.neg'

testX, testY = load_data_and_labels(positive_data_file, negative_data_file)


# In[9]:


save_dataset([trainX, trainY], 'train.pkl')
save_dataset([testX, testY], 'test.pkl')


# In[10]:


from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers import LSTM
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from pickle import load


# In[11]:


# load a clean dataset
def load_dataset(filename):
	return load(open(filename, 'rb'))
 
# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
 
# calculate the maximum document length
def max_length(lines):
	return len(lines)
 
# encode a list of lines
def encode_text(tokenizer, lines, length):
	# integer encode
	encoded = tokenizer.texts_to_sequences(lines)
	# pad encoded sequences
	padded = pad_sequences(encoded, maxlen = length, padding = 'post')
	return padded


# In[14]:


# define the model
def define_model(length, vocab_size):
	# channel 1
	input1 = Input(shape=(length,))
	embedding1 = Embedding(vocab_size, 18)(input1)
	conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
	drop1 = Dropout(0.5)(conv1)
	pool1 = MaxPooling1D(pool_size=2)(drop1)
	flat1 = Flatten()(pool1)
    
	# channel 2
	input2 = Input(shape=(length,))
	embedding2 = Embedding(vocab_size, 15)(input2)
	conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
	drop2 = Dropout(0.5)(conv2)
	pool2 = MaxPooling1D(pool_size=2)(drop2)
	flat2 = Flatten()(pool2)
    
	# channel 3
	input3 = Input(shape=(length,))
	embedding3 = Embedding(vocab_size, 12)(input3)
	conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling1D(pool_size=2)(drop3)
	flat3 = Flatten()(pool3)
    
	# merge
	CNN_layer = concatenate([flat1, flat2, flat3])
    
	# LSTM
	x = Embedding(vocab_size, 20)(input1)
	LSTM_layer = LSTM(128)(x)

	CNN_LSTM_layer = concatenate([LSTM_layer, CNN_layer])
    
	# interpretation
	dense1 = Dense(10, activation='relu')(CNN_LSTM_layer)
	outputs = Dense(1, activation='sigmoid')(dense1)
	model = Model(inputs=[input1, input2, input3], outputs=outputs)
    
	# compile
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
	# summarize
	print(model.summary())
	#plot_model(model, show_shapes=True, to_file='multichannel.png')
    
	return model


# In[ ]:


# load training dataset
trainLines, trainLabels = load_dataset('train.pkl')

# create tokenizer
tokenizer = create_tokenizer(trainLines)

# calculate max document length
trainLength = max_length(trainLines)

# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % trainLength)
print('Vocabulary size: %d' % vocab_size)

# encode data
trainX = encode_text(tokenizer, trainLines, trainLength)
print(trainX.shape)
 
# define model
model = define_model(trainLength, vocab_size)

# fit model
model.fit([trainX,trainX,trainX], array(trainLabels), epochs=1, batch_size=128)

# save the model
model.save('model.h5')


# In[80]:


from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
 
testLines, testLabels = load_dataset('test.pkl')
 
# create tokenizer
tokenizer = create_tokenizer(testLines)

# calculate max document length
length = trainLength

# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % length)
print('Vocabulary size: %d' % vocab_size)

# encode data
testX = encode_text(tokenizer, testLines, length)
print(testX.shape)

# load the model
model = load_model('model.h5') 

# evaluate model on test dataset dataset
loss, acc = model.evaluate([testX,testX,testX],array(testLabels), verbose=0)
print('Test accuracy: %f' % (acc*100))

