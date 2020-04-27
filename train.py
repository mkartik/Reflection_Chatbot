import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import pandas as pd

df = pd.read_csv('20200325_counsel_chat.csv')
df = df=df.dropna(subset=['answerText'])
df = df=df.dropna(subset=['questionTitle','questionText'])
train = df[df['split']=='train']
test = df[df['split']=='test']
val = df[df['split']=='val']

words=[]
topics = []
documents = []
ignore_words = ['?', '!']
lemmatizer = WordNetLemmatizer()

for index,rows in train.iterrows():
    sentence  = None
    if (not pd.isnull(rows['questionText'])):
        sentence = rows['questionText']
    else:
        sentence = rows['questionTitle']
    w = nltk.word_tokenize(sentence)
    words.extend(w)
    documents.append((w, rows['topic']))

    if rows['topic'] not in topics:
        topics.append(rows['topic'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
topics = sorted(list(set(topics)))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(topics,open('topics.pkl','wb'))

training = []
output_empty = [0] * len(topics)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[topics.index(doc[1])] = 1
    training.append([bag, output_row])
    
random.shuffle(training)
training = np.array(training)
train_x = list(training[:,0])
train_y = list(training[:,1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=250, batch_size=5, verbose=1)
model.save('emp_chatbot_model.h5', hist)