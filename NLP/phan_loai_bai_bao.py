from pyvi import ViTokenizer, ViPosTagger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import gensim
import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords  # lib for remove stop word
import pickle
# data precessing
label = []
describe_train = []
data_train = []
data_test = []
describe_test = []

with open('trainning.txt','r',encoding='utf-8') as f:
    lines_train = f.read()
    data_train = lines_train.split('\n')
f.close()

with open('testing.txt','r',encoding='utf-8') as f_test:
    line_test = f_test.read()
    data_test = line_test.split('\n')

# tách dữ liệu ghép vào từng mảng
#train data
n_train = len(data_train)
for i in range(1,n_train - 1):
    des = data_train[i].split('\t')
    label.append(des[0])
    des[1] = gensim.utils.simple_preprocess(des[1].lower())
    str = ""
    for i in des[1]:
        str = str + i + " "
    des[1] = ViTokenizer.tokenize(str)
    describe_train.append(remove_stopwords(des[1]))

# test data
for i in data_test:
    des = gensim.utils.simple_preprocess(i.lower())
    str = ""
    for i in des:
        str = str + i + " "
    des = ViTokenizer.tokenize(str)
    describe_test.append(remove_stopwords(des))

pickle.dump(describe_train,open('train.pkl','wb'))
pickle.dump(describe_test,open('test.pkl','wb'))
pickle.dump(label,open('label.pkl','wb'))