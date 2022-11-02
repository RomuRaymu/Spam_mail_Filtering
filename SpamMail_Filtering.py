# %%
# Import Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urlib.request

from sklearn.mode_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import backend as K

# %%
# 스팸 메일 csv Download
spam_data = pd.read_csv('drive/MyDrive/KIS_KIS.csv')

# %%
# SNS 대화 내역 json Download
# 비스팸 메일 역할
# 하나의 대화 내역 안에는 1번 화자와 2번 화자가 존재
# 1번 화자가 적은 메세지들을 1개의 sentence 로 재구성
# 2번 화자가 적은 메세지들을 1개의 sentence 로 재구성

import json

with open('drive/MyDrive/sample.json', 'r') as j:
  chat_contents = json.loads(j.read()) # Json read

chat1 = '' # 1번 화자가 적은 메세지들을 합친 문장
chat2 = '' # 2번 화자가 적은 메세지들을 합친 문장
ham_list = [] # 모든 문장 list

for i in range(len(chat_contents['data'])):
  for j in range(len(chat_contents['data'][i]['body']['dialogue'])):
    if chat_contents['data'][i]['body']['dialogue'][j]['participantID'] == 'P01':
      chat1 += chat_contents['data'][j]['body']['dialogue'][i]['utterance']
      chat1 += ' '
    else:
      P02 += chat_contents['data'][j]['body']['dialogue'][i]['utterance']
      P02 += ' '
  ham_list.append(P01)
  ham_list.append(P02)
  P01 = ''
  P02 = ''

# %%
# 비스팸 Data 정리
ham = pd.DataFrame({'Text':ham_list}) # DataFrame 화
ham.insert(0, 'Spam', 0) # Spam 카테고리 추가 (Ham 을 뜻하는 0)

# %%
# 스팸 Data 정리
# 스팸 Data 를 최종적으로 ['Spam', 'Text'] 형태로 변경
del spam_data['2020']
del spam_data['01']
del spam_data['01.1']
del spam_data['090100']
del spam_data['***********']
del spam_data['***********.1'] # Text 제외 모든 값 제거
spam_data.columns = ['Text'] # Text 의 Header 추가
spam_data.insert(0, 'Spam', 1) # Spam 구분값 추가 (Spam 을 뜻하는 1)

# Text 열에서 중복인 내용이 있다면 제거
spam_data.drop_duplicates(subset=['Text'], inplace=True)

# Ham Data 와 갯수를 맞추기 위해 일부 갯수만 추출
spam_test_data = spam_data.sample(n=len(ham))

# %%
# Spam Data 와 Ham Data 병합
dataSet = pd.concat([spam_data, ham]) # concat
dataSet = dataSet.sample(frac=1).rest_index(drop=True) # shuffling & index reset

# 한글과 숫자만 남기고, 나머지 글자들은 공백으로 변경
dataSet['Text'] = dataSet['Text'].str.replace('[^ㄱ-ㅎ ㅏ-ㅣ 가-힣 0-9]', ' ')

# %%
# 중복 Text 제거
dataSet.drop_duplicates(subset=['Text'], inplace=True)

print('총 샘플의 수 :', len(dataSet))

# %%
# soynlp 단어 분리기 모듈 import
# !pip install soynlp

from soynlp.word import WordExtractor
from soynlp.tokenizer import MaxScoreTokenizer
from soynlp.tokenizer import LTokenizer

# %%
word_extractor = WordExtractor()
word_extractor.train(texts['Text']) # Train 그룹의 Text 를 단어로 분리하기 위한 훈련
word_score_table = word_extractor.extract()
cohesion_score = {word:score.cohesion_forward for word, score in word_score_table.items()}
maxscore_tokenizer = MaxScoreTokenizer(scores=cohesion_score)

def soynlp_morphs(contents):
  return ' '.join(maxscore_tokenizer.tokenize(contents)) # 

texts['soynlp_Text'] = texts['Text'].apply(soynlp_morphs)

# %%
# 변환한 문장 중에 중복인 내용 제거
texts.drop_duplicates(subset=['soynlp_Text'], inplace=True)

# %%
# Data 와 Lable 분리 저장
X_data = dataSet['soynlp_Text']
y_data = dataSet['Spam']

# %%
# Train 용 Data, Test 용 Data 분리
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0, stratify=y_data)

# %%
# Keras Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train) # Tokenizer 에 단어 dictonary 저장
X_train_encoded = tokenizer.texts_to_sequences(X_train) # Text 숫자화

# %%
word_to_index = tokenizer.word_index # 단어

threshold = 2
total_cnt = len(word_to_index) #  단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold 보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold 보다 작은 단어의 등장 빈도수의 총 

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
  total_freq = total_freq + value

  # 단어의 등장 빈도수가 threshold보다 작으면
  if(value < threshold):
    rare_cnt = rare_cnt + 1
    rare_freq = rare_freq + value

print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print('단어 집합(vocabulary)에서 희귀 단어의 비율:', (rare_cnt / total_cnt) * 100)
print('전체 등장 빈도에서 희귀 단어 등장 빈도 비율:', (rare_freq / total_freq) * 100)

# %%
vocab_size = len(word_to_index) + 1 # 정리한 후의 단어 종류 갯수

# %%
print('메일의 최대 길이 : %d' % max(len(sample) for sample in X_train_encoded))
print('메일의 평균 길이 : %f' % (sum(map(len, X_train_encoded))/len(X_train_encoded)))

max_len = (int)(sum(map(len, X_train_encoded))/len(X_train_encoded)) + 1 # Text의 최대 길이
X_train_padded = pad_sequences(X_train_encoded, maxlen = max_len) # max_len 크기만큼 모든 Text 의 길이 증가 / 증가 된 공간에는 0 추가

# %%
# recall, precision, f1score 을 구하는 함수 https://m.blog.naver.com/wideeyed/221226716255
def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall


def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
    count_true_positive_false_positive = K.sum(y_pred_yn)

    # Precision = (True Positive) / (True Positive + False Positive)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # return a single tensor value
    return precision


def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    _f1score = ( 2 * _recall * _precision) / (_recall + _precision+ K.epsilon())
    
    # return a single tensor value
    return _f1score

# %%
# 훈련 model 생성 및 훈련
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential

embedding_dim = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_shape=[max_len]))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(8))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc', precision, recall, f1score]) # 위에 설정한 함수를 '' 없이 직접 적음.
history = model.fit(X_train_padded, y_train, epochs=5, batch_size=64, validation_split=0.2)

# %%
model.summary()

# %%
# Test 데이터로 확인
X_test_encoded = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_encoded, maxlen = max_len)
data = model.evaluate(X_test_padded, y_test)
print("\n 테스트 정확도: %.4f" % (data[1]))
print(" 테스트 정밀도: %.4f" % (data[2]))
print(" 테스트 재현율: %.4f" % (data[3]))
print(" 테스트 F1 Score: %.4f" % (data[4]))

# %%
# 그래프로 설명
epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.title('model accuracy')
plt.plot(epochs, history.history['acc'])
plt.plot(epochs, history.history['precision'])
plt.plot(epochs, history.history['recall'])
plt.plot(epochs, history.history['f1score'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc', 'precision', 'recall', 'f1score'], loc='lower right')
plt.show()