import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import tweepy
import re
import nltk
from tweepy import OAuthHandler
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import text_to_word_sequence

# define the document
# Loading training/labeled dataset
data = []
data = pd.read_csv('~/Downloads/training.csv', header = None, sep = ',', encoding = 'latin-1', error_bad_lines = False)
tweets = data[1].values.tolist()
sent = data[0].values.tolist()
tweets = np.asarray(tweets)

# Clean data
# Lemitization
## Should this be done? effects?
lemmatizer=WordNetLemmatizer()
for i in range(len(tweets)):
    words1 = text_to_word_sequence(tweets[i])
    newwords1=[]
    for j in words1:
        j = lemmatizer.lemmatize(j)
        newwords1.append(j)
    tweets[i]=' '.join(newwords1)

#removing stop words
## Should this be done? will stop words not provide order to the sentence and will that order not matter when fed to the first layer of the NN?
from nltk.corpus import stopwords

word_tokens=[]
stop_words = set(stopwords.words('english'))

for i in range(len(tweets)):
    words=text_to_word_sequence(tweets[i])
    newwords=[word for word in words if word not in stop_words]
    tweets[i]=' '.join(newwords)


# tokenize the new cleaned individual tweets
token_tweets = []
for i in range(len(tweets)):
    token_tweets.append(text_to_word_sequence(tweets[i]))


# estimate the size of the vocabulary
# Create corpus of tweets/merge all tweets into single document
doc = ''
for i in range(len(tweets)):
    doc = doc + ' ' + tweets[i]

dict = set(text_to_word_sequence(doc))
len(dict)

# integer encode the document
from keras.preprocessing.text import hashing_trick
int_tweets = []
for i in range(len(tweets)):
    int_tweets.append(hashing_trick(tweets[i], round(len(dict)*1.3), hash_function='md5'))

# Normalizing tweets length / Padding
# Creating tensors for NN
tensor_tweets = keras.preprocessing.sequence.pad_sequences(int_tweets, value = round(len(dict)*1.3)+1, padding = 'post', maxlen=30)

## Creating training and testing set
indices = np.random.permutation(tensor_tweets.shape[0])
training_idx, test_idx = indices[:10000], indices[10000:]
trainingx, testx = tensor_tweets[training_idx], tensor_tweets[test_idx]

x_val = testx
partial_x_train = trainingx

sent = np.asarray(sent)
sent = np.transpose(sent)
trainingy, testy = sent[training_idx], sent[test_idx]

y_val = testy
partial_y_train = trainingy


# Build Model
# input shape is the vocabulary count used for the movie reviews (10,000 words)
model = keras.Sequential()
model.add(keras.layers.Embedding(round(len(dict)*1.3)+2, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=200,
                    batch_size=500,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(tensor_tweets[:20000], sent[:20000])

print(results)

history_dict = history.history
history_dict.keys()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)


import plotly.plotly as py
import plotly.graph_objs as go

# Create a trace
trace0 = go.Scatter(
    x = np.array(epochs),
    y = acc,
    mode = 'markers'
)

trace1 = go.Scatter(
    x = np.array(epochs),
    y = val_acc,
    mode = 'lines'
)

trace2 = go.Scatter(
    x = np.array(epochs),
    y = loss,
    mode = 'lines'
)

trace3 = go.Scatter(
    x = np.array(epochs),
    y = val_loss,
    mode = 'lines'
)


graphdata1 = [trace0, trace1]
graphdata2 = [trace2, trace3]

# Plot and embed in ipython notebook!
#py.plot(graphdata1, filename='basic-scatter')
#py.plot(graphdata2, filename='basic_scatter1')

