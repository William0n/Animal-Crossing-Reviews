import pandas as pd
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('AC data.csv', encoding='cp1252')


data = data.drop(columns =['user_name','date'])
data.describe()
data['grade'].astype(object)

### Splitting data into high & low ratings
rating_count = data['grade'].value_counts().sort_index()
low_rating_sum = rating_count.iloc[:5].sum()
high_rating_sum = rating_count.iloc[5:].sum()

data['grade'] = np.where(data['grade'] > 4,  1, 0)
data = data.to_numpy()

train_set, test_set = train_test_split(data,test_size = 0.3, random_state = 10)

x_train = train_set[:, 1]
y_train = train_set[:, 0]

x_test = test_set[:,1]
y_test = test_set[:,0]

### Preprocess function 

def preprocess(x_batch): 
    x_batch = tf.strings.substr(x_batch, 0, 400)
    x_batch = tf.strings.regex_replace(x_batch, b'<br\\s*/?>', b" ")
    x_batch = tf.strings.regex_replace(x_batch, b"[^a-zA-Z']", b" ")
    x_batch = tf.strings.split(x_batch)
    return x_batch.to_tensor(default_value = b'<pad>') 
l = data[56,1]
tf.strings.substr(l, 0 ,250)

x_train = preprocess(x_train)

from collections import Counter
dictionary = Counter()
for review in x_train:
    dictionary.update(list(review.numpy()))
    
dictionary.most_common()[:10]

### Encoding text 

vocab_size = 2000
truncated_dict = [word for word, count in dictionary.most_common()[:vocab_size]]

words = tf.constant(truncated_dict)
word_index = tf.range(len(truncated_dict), dtype= tf.int64)

vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_index)
oov_bucket = 500
table = tf.lookup.StaticVocabularyTable(vocab_init, oov_bucket)

def encode_text(x_batch):
    return(table.lookup(x_batch))
        
x_train = encode_text(x_train)

y_train = np.asarray(y_train, dtype = np.int64)
y_train = tf.convert_to_tensor(y_train)


x_test = preprocess(x_test)
x_test = encode_text(x_test)

y_test = np.asarray(y_test, dtype = np.int64)
y_test = tf.convert_to_tensor(y_test)


### Modeling 

# First arg in embed layer is the # of distinct words. Second arg is the dim of the weights...kinda 
# like a dimension reduction 

embed_size = 150
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size + oov_bucket, embed_size, input_shape = [None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences = True)),
    tf.keras.layers.GRU(128),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

lstm_model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])


history = lstm_model.fit(x_train, y_train, epochs = 10, batch_size = 30, 
                         validation_split = 0.10)
test_results = lstm_model.evaluate(x_test, y_test)

### Training accuracy and loss graphs

plt.figure(figsize = (14,8))
plt.subplot(1,2,1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['training', 'validation'])

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['training', 'validation'])
plt.show()


### Customized test review to see model performance

r = tf.constant([b'This game is really really bad game'])
r= preprocess(r)
r = encode_text(r)
sample_predict = lstm_model.predict(r)[0][0] * 100

print('This game review is ' f"{round(sample_predict, ndigits = 2)}% positive")





