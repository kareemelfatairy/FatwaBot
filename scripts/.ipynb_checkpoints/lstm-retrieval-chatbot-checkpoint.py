import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from six.moves import cPickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.stem import ISRIStemmer
from keras.utils import np_utils
import tensorflow as tf
from keras import backend as K

BATCH_SIZE = 16 # Batch size for GPU
NUM_WORDS = 10000 # Vocab length
MAX_LEN = 20 # Padding length (# of words)
LSTM_EMBED = 8 # Number of LSTM nodes

K.set_learning_phase(False)

data = pd.read_csv('../dataset/ASKFM-master/full_dataset.csv')
tokenizer = cPickle.load(open("../models/lstm-autoencoder-tokenizer.pickle", "rb"))

stemmer = ISRIStemmer()

stemmer = ISRIStemmer()

# Read the encoder model
model = tf.keras.models.load_model('../models/lstm25/lstm-encoder.h5',compile=False)
model.load_weights('../models/lstm_encoder_weights.h5')
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')


# Create the encoding function
encode = K.function([model.input, K.learning_phase()], [model.layers[1].output])

Questions = tokenizer.texts_to_sequences(data.Question)
# We pad sequences that are shorter than MAX_LEN
Questions = pad_sequences(Questions, padding='post', truncating='post', maxlen=MAX_LEN)
Questions = np.squeeze(np.array(encode([Questions])))
while True:
    question = input('Please enter a question: \n')
    question = stemmer.stem(question)
    question = tokenizer.texts_to_sequences([question])
    question = pad_sequences(question, padding='post', truncating='post', maxlen=MAX_LEN)
    question = np.squeeze(encode([question]))

    rank = cosine_similarity(question.reshape(1, -1), Questions)
    top = np.argsort(rank, axis=-1).T[-5:].tolist()
    for item in top:
        print(data['Answer'].iloc[item].values[0])
