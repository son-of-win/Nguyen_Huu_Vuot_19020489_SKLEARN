from pyvi import ViTokenizer, ViPosTagger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import gensim
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.parsing.preprocessing import remove_stopwords  # lib for remove stop word
import pickle
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

train_data = pickle.load(open('train.pkl','rb'))
test_data = pickle.load(open('test.pkl','rb'))
label = pickle.load(open('label.pkl','rb'))

encoder = LabelEncoder()
label = encoder.fit_transform(label)


##############
with open('testing.txt','r',encoding='utf-8') as f_test:
    line_test = f_test.read()
    data_test = line_test.split('\n')

# sử dụng Tf-Idf vector để tạo một ma trận từ điển
# trong đó mỗi hàng sẽ đại diện cho một văn bản, mỗi cột sẽ đại diện cho một từ trong từ điển
# mỗi ô giá trị sẽ là một giá trị đại diện cho tần xuất xuất hiện của từ tương ứng
tfidf_vector = TfidfVectorizer(analyzer='word',max_features=30000)
tfidf_vector.fit(train_data)

train_data_tfidf = tfidf_vector.transform(train_data)
test_data_tfidf = tfidf_vector.transform(test_data)
#
# ### Xây dựng mô hình phân lớp
data_train ,  data_test , label_train,label_test = train_test_split(train_data_tfidf, label, test_size=0.2, random_state=0)

# classifier with native bayes
# classifier = MultinomialNB()
# classifier.fit(data_train,label_train)
# label_fact, label_predict = label_test, classifier.predict(data_test)
# print(classification_report(label_fact,label_predict))

#classifier with linear regression
linear = LogisticRegression()
linear.fit(data_train,label_train)
label_fact, label_predict_linear = label_test, linear.predict(data_test)
print(classification_report(label_fact,label_predict_linear))
result_label = linear.predict(test_data_tfidf)
# with open('result.txt','w',encoding='utf-16') as file:
#     for a,b in zip(result_label,data_test):
#
# file.close()