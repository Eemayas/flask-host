from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
import pickle

commentCountPerPage = 12

# Load Keras model
# lstm = load_model('app/api/flak/Model/LSTM_sentimentmodel.h5')
rnn = load_model('./api/Model/RNN_sentimentmodel.h5')
# gru = load_model('app/api/flask/Model/GRU_sentimentmodel.h5')
# rnn = load_model('app/api/flask/rnnmodel.h5')


tokenizer_LSTM = Tokenizer()
with open('./api/Tokenizer/LSTMtokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer_LSTM = pickle.load(tokenizer_file)

tokenizer_RNN = Tokenizer()
with open('./api/Tokenizer/RNNtokenizer.pkl', 'rb') as tokenizer_file_RNN:
    tokenizer_RNN = pickle.load(tokenizer_file_RNN)
