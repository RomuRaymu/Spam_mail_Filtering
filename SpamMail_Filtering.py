{"nbformat":4,"nbformat_minor":0,"metadata":{"colab":{"provenance":[],"collapsed_sections":[],"authorship_tag":"ABX9TyOsOYv81TuvvbjXH2vYEX+p"},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},"cells":[{"cell_type":"code","execution_count":null,"metadata":{"id":"lyLHo-7J88TW"},"outputs":[],"source":["# Import Modules\n","import numpy as np\n","import pandas as pd\n","import matplotlib.pyplot as plt\n","import urlib.request\n","\n","from sklearn.mode_selection import train_test_split\n","from tensorflow.keras.preprocessing.text import Tokenizer\n","from tensorflow.keras.preprocessing.sequence import pad_sequences\n","from keras import backend as K"]},{"cell_type":"code","source":["# 스팸 메일 csv Download\n","spam_data = pd.read_csv('drive/MyDrive/KIS_KIS.csv')"],"metadata":{"id":"Pe5sbwwY9VgT","executionInfo":{"status":"ok","timestamp":1667050612471,"user_tz":-540,"elapsed":3,"user":{"displayName":"초콜릿반지","userId":"03689539703216079168"}}},"execution_count":1,"outputs":[]},{"cell_type":"code","source":["# SNS 대화 내역 json Download\n","# 비스팸 메일 역할\n","# 하나의 대화 내역 안에는 1번 화자와 2번 화자가 존재\n","# 1번 화자가 적은 메세지들을 1개의 sentence 로 재구성\n","# 2번 화자가 적은 메세지들을 1개의 sentence 로 재구성\n","\n","import json\n","\n","with open('drive/MyDrive/sample.json', 'r') as j:\n","  chat_contents = json.loads(j.read()) # Json read\n","\n","chat1 = '' # 1번 화자가 적은 메세지들을 합친 문장\n","chat2 = '' # 2번 화자가 적은 메세지들을 합친 문장\n","ham_list = [] # 모든 문장 list\n","\n","for i in range(len(chat_contents['data'])):\n","  for j in range(len(chat_contents['data'][i]['body']['dialogue'])):\n","    if chat_contents['data'][i]['body']['dialogue'][j]['participantID'] == 'P01':\n","      chat1 += chat_contents['data'][j]['body']['dialogue'][i]['utterance']\n","      chat1 += ' '\n","    else:\n","      P02 += chat_contents['data'][j]['body']['dialogue'][i]['utterance']\n","      P02 += ' '\n","  ham_list.append(P01)\n","  ham_list.append(P02)\n","  P01 = ''\n","  P02 = ''"],"metadata":{"id":"qTm5STE99Ybg"},"execution_count":null,"outputs":[]},{"cell_type":"code","source":["# 비스팸 Data 정리\n","ham = pd.DataFrame({'Text':ham_list}) # DataFrame 화\n","ham.insert(0, 'Spam', 0) # Spam 카테고리 추가 (Ham 을 뜻하는 0)"],"metadata":{"id":"hBVVgmEq_C8x"},"execution_count":null,"outputs":[]},{"cell_type":"code","source":["# 스팸 Data 정리\n","# 스팸 Data 를 최종적으로 ['Spam', 'Text'] 형태로 변경\n","del spam_data['2020']\n","del spam_data['01']\n","del spam_data['01.1']\n","del spam_data['090100']\n","del spam_data['***********']\n","del spam_data['***********.1'] # Text 제외 모든 값 제거\n","spam_data.columns = ['Text'] # Text 의 Header 추가\n","spam_data.insert(0, 'Spam', 1) # Spam 구분값 추가 (Spam 을 뜻하는 1)\n","\n","# Text 열에서 중복인 내용이 있다면 제거\n","spam_data.drop_duplicates(subset=['Text'], inplace=True)\n","\n","# Ham Data 와 갯수를 맞추기 위해 일부 갯수만 추출\n","spam_test_data = spam_data.sample(n=len(ham))"],"metadata":{"id":"kSJcwK8D_lWa","executionInfo":{"status":"ok","timestamp":1667051410461,"user_tz":-540,"elapsed":2,"user":{"displayName":"초콜릿반지","userId":"03689539703216079168"}}},"execution_count":2,"outputs":[]},{"cell_type":"code","source":["# Spam Data 와 Ham Data 병합\n","dataSet = pd.concat([spam_data, ham]) # concat\n","dataSet = dataSet.sample(frac=1).rest_index(drop=True) # shuffling & index reset\n","\n","# 한글과 숫자만 남기고, 나머지 글자들은 공백으로 변경\n","dataSet['Text'] = dataSet['Text'].str.replace('[^ㄱ-ㅎ ㅏ-ㅣ 가-힣 0-9]', ' ')"],"metadata":{"id":"BYTo4THKAdC3"},"execution_count":null,"outputs":[]},{"cell_type":"code","source":["# 중복 Text 제거\n","dataSet.drop_duplicates(subset=['Text'], inplace=True)\n","\n","print('총 샘플의 수 :', len(dataSet))"],"metadata":{"id":"-Vzx9K1WCc2w"},"execution_count":null,"outputs":[]},{"cell_type":"code","source":["# soynlp 단어 분리기 모듈 import\n","!pip install soynlp\n","\n","from soynlp.word import WordExtractor\n","from soynlp.tokenizer import MaxScoreTokenizer\n","from soynlp.tokenizer import LTokenizer"],"metadata":{"id":"VEbi2oQvJSaX"},"execution_count":null,"outputs":[]},{"cell_type":"code","source":["word_extractor = WordExtractor()\n","word_extractor.train(texts['Text']) # Train 그룹의 Text 를 단어로 분리하기 위한 훈련\n","word_score_table = word_extractor.extract()\n","cohesion_score = {word:score.cohesion_forward for word, score in word_score_table.items()}\n","maxscore_tokenizer = MaxScoreTokenizer(scores=cohesion_score)\n","\n","def soynlp_morphs(contents):\n","  return ' '.join(maxscore_tokenizer.tokenize(contents)) # \n","\n","texts['soynlp_Text'] = texts['Text'].apply(soynlp_morphs)"],"metadata":{"id":"2qLGAcOOXsu7"},"execution_count":null,"outputs":[]},{"cell_type":"code","source":["# 변환한 문장 중에 중복인 내용 제거\n","texts.drop_duplicates(subset=['soynlp_Text'], inplace=True)"],"metadata":{"id":"CctutUwIX7lF"},"execution_count":null,"outputs":[]},{"cell_type":"code","source":["# Data 와 Lable 분리 저장\n","X_data = dataSet['soynlp_Text']\n","y_data = dataSet['Spam']"],"metadata":{"id":"qgXlGBplCzh1"},"execution_count":null,"outputs":[]},{"cell_type":"code","source":["# Train 용 Data, Test 용 Data 분리\n","X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0, stratify=y_data)"],"metadata":{"id":"wBcAfxXkZe4C"},"execution_count":null,"outputs":[]},{"cell_type":"code","source":["# Keras Tokenizer\n","tokenizer = Tokenizer()\n","tokenizer.fit_on_texts(X_train) # Tokenizer 에 단어 dictonary 저장\n","X_train_encoded = tokenizer.texts_to_sequences(X_train) # Text 숫자화"],"metadata":{"id":"C1Sm-JS0Z2kA"},"execution_count":null,"outputs":[]},{"cell_type":"code","source":["word_to_index = tokenizer.word_index # 단어\n","\n","threshold = 2\n","total_cnt = len(word_to_index) #  단어의 수\n","rare_cnt = 0 # 등장 빈도수가 threshold 보다 작은 단어의 개수를 카운트\n","total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합\n","rare_freq = 0 # 등장 빈도수가 threshold 보다 작은 단어의 등장 빈도수의 총 \n","\n","# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.\n","for key, value in tokenizer.word_counts.items():\n","  total_freq = total_freq + value\n","\n","  # 단어의 등장 빈도수가 threshold보다 작으면\n","  if(value < threshold):\n","    rare_cnt = rare_cnt + 1\n","    rare_freq = rare_freq + value\n","\n","print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))\n","print('단어 집합(vocabulary)에서 희귀 단어의 비율:', (rare_cnt / total_cnt) * 100)\n","print('전체 등장 빈도에서 희귀 단어 등장 빈도 비율:', (rare_freq / total_freq) * 100)"],"metadata":{"id":"z_QdGjhnaFSX"},"execution_count":null,"outputs":[]},{"cell_type":"code","source":["vocab_size = len(word_to_index) + 1 # 정리한 후의 단어 종류 갯수"],"metadata":{"id":"TAMW4sZhhsBT"},"execution_count":null,"outputs":[]},{"cell_type":"code","source":["print('메일의 최대 길이 : %d' % max(len(sample) for sample in X_train_encoded))\n","print('메일의 평균 길이 : %f' % (sum(map(len, X_train_encoded))/len(X_train_encoded)))\n","\n","max_len = (int)(sum(map(len, X_train_encoded))/len(X_train_encoded)) + 1 # Text의 최대 길이\n","X_train_padded = pad_sequences(X_train_encoded, maxlen = max_len) # max_len 크기만큼 모든 Text 의 길이 증가 / 증가 된 공간에는 0 추가"],"metadata":{"id":"jPVv4o1qjT3l"},"execution_count":null,"outputs":[]},{"cell_type":"code","source":["# recall, precision, f1score 을 구하는 함수 https://m.blog.naver.com/wideeyed/221226716255\n","def recall(y_target, y_pred):\n","    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다\n","    # round : 반올림한다\n","    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다\n","    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다\n","\n","    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다\n","    count_true_positive = K.sum(y_target_yn * y_pred_yn) \n","\n","    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체\n","    count_true_positive_false_negative = K.sum(y_target_yn)\n","\n","    # Recall =  (True Positive) / (True Positive + False Negative)\n","    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다\n","    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())\n","\n","    # return a single tensor value\n","    return recall\n","\n","\n","def precision(y_target, y_pred):\n","    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다\n","    # round : 반올림한다\n","    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다\n","    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다\n","\n","    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다\n","    count_true_positive = K.sum(y_target_yn * y_pred_yn) \n","\n","    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체\n","    count_true_positive_false_positive = K.sum(y_pred_yn)\n","\n","    # Precision = (True Positive) / (True Positive + False Positive)\n","    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다\n","    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())\n","\n","    # return a single tensor value\n","    return precision\n","\n","\n","def f1score(y_target, y_pred):\n","    _recall = recall(y_target, y_pred)\n","    _precision = precision(y_target, y_pred)\n","    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다\n","    _f1score = ( 2 * _recall * _precision) / (_recall + _precision+ K.epsilon())\n","    \n","    # return a single tensor value\n","    return _f1score"],"metadata":{"id":"MsshTeHljkn7"},"execution_count":null,"outputs":[]},{"cell_type":"code","source":["# 훈련 model 생성 및 훈련\n","from tensorflow.keras.layers import Embedding, Dense, LSTM\n","from tensorflow.keras.models import Sequential\n","\n","embedding_dim = 128\n","\n","model = Sequential()\n","model.add(Embedding(vocab_size, embedding_dim, input_shape=[max_len]))\n","model.add(LSTM(128, return_sequences=True))\n","model.add(LSTM(32, return_sequences=True))\n","model.add(LSTM(8))\n","model.add(Dense(1, activation='sigmoid'))\n","\n","model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc', precision, recall, f1score]) # 위에 설정한 함수를 '' 없이 직접 적음.\n","history = model.fit(X_train_padded, y_train, epochs=5, batch_size=64, validation_split=0.2)"],"metadata":{"id":"mLfwKCPGjnso"},"execution_count":null,"outputs":[]},{"cell_type":"code","source":["model.summary()"],"metadata":{"id":"Wf3xyf8VjpRq"},"execution_count":null,"outputs":[]},{"cell_type":"code","source":["# Test 데이터로 확인\n","X_test_encoded = tokenizer.texts_to_sequences(X_test)\n","X_test_padded = pad_sequences(X_test_encoded, maxlen = max_len)\n","data = model.evaluate(X_test_padded, y_test)\n","print(\"\\n 테스트 정확도: %.4f\" % (data[1]))\n","print(\" 테스트 정밀도: %.4f\" % (data[2]))\n","print(\" 테스트 재현율: %.4f\" % (data[3]))\n","print(\" 테스트 F1 Score: %.4f\" % (data[4]))"],"metadata":{"id":"Qt6f3c3pj1DG"},"execution_count":null,"outputs":[]},{"cell_type":"code","source":["# 그래프로 설명\n","epochs = range(1, len(history.history['acc']) + 1)\n","plt.plot(epochs, history.history['loss'])\n","plt.plot(epochs, history.history['val_loss'])\n","plt.title('model loss')\n","plt.ylabel('loss')\n","plt.xlabel('epoch')\n","plt.legend(['train', 'val'], loc='upper left')\n","plt.show()\n","\n","plt.title('model accuracy')\n","plt.plot(epochs, history.history['acc'])\n","plt.plot(epochs, history.history['precision'])\n","plt.plot(epochs, history.history['recall'])\n","plt.plot(epochs, history.history['f1score'])\n","plt.ylabel('accuracy')\n","plt.xlabel('epoch')\n","plt.legend(['acc', 'precision', 'recall', 'f1score'], loc='lower right')\n","plt.show()"],"metadata":{"id":"C0HzITcuj29j"},"execution_count":null,"outputs":[]}]}