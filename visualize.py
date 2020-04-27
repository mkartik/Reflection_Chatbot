import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd
import random
from sentence_transformers import SentenceTransformer
from keras.models import load_model
import nltk.translate.bleu_score as bleu
import nltk.translate.gleu_score as gleu

model = load_model('emp_chatbot_model.h5')
model_st = SentenceTransformer('bert-base-nli-mean-tokens')
lemmatizer = WordNetLemmatizer()

df = pd.read_csv('20200325_counsel_chat.csv')
df = df=df.dropna(subset=['answerText'])
df = df=df.dropna(subset=['questionTitle','questionText'])
train = df[df['split']=='train']

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('topics.pkl','rb'))
words_reflec = ['It seems like you', 'It sound like you', 'I feel that you']

def square_rooted(x):
    return np.sqrt(sum([a*a for a in x]))

def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return numerator/float(denominator)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"topic": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, train, msg):
    tag = ints[0]['topic']
    sents = df[df['topic']==tag]['questionText'].values
    se = model_st.encode(sents)
    se_msg = model_st.encode([msg])
    
    max_val = 0
    pos = 0
    for p,i in enumerate(se):
        cos = cosine_similarity(se_msg[0], i)
        if max_val<cos:
            max_val = cos
            pos = p
    
    res = df[df['topic']==tag].iloc[pos]
    result = res['answerText'].split('.')[0]
    # print ('dataset response: ', result)
    # result = random.choice(words_reflec) + ' are facing ' + tag
    return result

def chatbot_response(msg):
    ids = predict_class(msg, model)
    res = getResponse(ids, train, msg)
    return res

def bleu_score(msg,res):
    smoothing_func = nltk.translate.bleu_score.SmoothingFunction()
    score = bleu.sentence_bleu(msg, res, smoothing_function=smoothing_func.method0)
    return score

msg = "I am not happy with my job"
# msg = "I"
res = chatbot_response(msg)
print (res)




