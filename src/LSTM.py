import gensim
import numpy as np
from emo_utils import *
import emoji
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras import optimizers
import matplotlib.pyplot as plt

np.random.seed(1)

WS = pd.read_excel('Data/labeled-chat-data.xlsx', header=None)
WS_np = np.array(WS)
Y = WS_np[:, 0]
X = WS_np[:, 1]
Y = Y.astype(int)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

maxLen = 0
for i in range(len(X)):
    A = X[i].split()
    if len(A) > maxLen:
        maxLen = len(A)

Y_oh_train = convert_to_one_hot(Y_train, C=3)
Y_oh_test = convert_to_one_hot(Y_test, C=3)

model_gensim = gensim.models.Word2Vec.load('Data/embedding')
model = Word2Vec.load('Data/embedding')

word_to_index = {}
index_to_word = {}
for index in range(2381122):
    word = model.wv.index2word[index]
    word_to_index[word] = index
    index_to_word[index] = word

word_to_vec_map = {}
for index in range(2381122):
    word = model.wv.index2word[index]
    word_to_vec_map[word] = model.wv[word]


def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))

    for i in range(m):

        sentence_words = X[i].lower().split()

        j = 0

        for w in sentence_words:
            if w in word_to_index:
                X_indices[i, j] = word_to_index[w]

            j = j + 1

    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map["cucumber"].shape[0]

    emb_matrix = np.zeros((vocab_len, emb_dim))

    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]

    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

    embedding_layer.build((None,))

    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)


def TWITCH_MODEL(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = Input(input_shape, dtype='int32')

    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    embeddings = embedding_layer(sentence_indices)

    X = LSTM(128, return_sequences=True)(embeddings)

    X = Dropout(0.3)(X)
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(3)(X)
    X = Activation('softmax')(X)

    model = Model(inputs=sentence_indices, outputs=X)

    return model


if __name__ == '__main__':
    model = TWITCH_MODEL((maxLen,), word_to_vec_map, word_to_index)
    model.summary()

    optimizer = optimizers.Adam(decay=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
    Y_train_oh = convert_to_one_hot(Y_train, C=3)

    X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=maxLen)
    Y_test_oh = convert_to_one_hot(Y_test, C=3)

    model.fit(X_train_indices, Y_train_oh, epochs=50, validation_data=(X_test_indices, Y_test_oh), batch_size=32,
              shuffle=True)

