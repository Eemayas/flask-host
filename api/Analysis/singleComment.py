from flask import jsonify, request
from flask import jsonify
import numpy as np

from constants import tokenizer_RNN, rnn
from keras.preprocessing.sequence import pad_sequences
from flask import request, jsonify
from preprocessing import removeemoji, filter_english_comments, preprocessing_RNN


def single_comment_analysis():
    comment = request.args.get('text')
    initComment = comment
    print(f"comment {comment}")
    if comment is None:
        return jsonify({"error": "Failed to retrieve Text"}), 500
    comment = removeemoji(comment)
    print(f"emoji {comment}")
    comment = filter_english_comments(comment)
    print(f"filter {comment}")

    comment = preprocessing_RNN(comment)
    if not comment or comment == "" or comment == ".":
        return jsonify({"error": "Text is not in english Language"}), 500
    else:

        sequence_RNN = tokenizer_RNN.texts_to_sequences([comment])
        padded_sequences_RNN = pad_sequences(sequence_RNN, maxlen=100)
        prediction_RNN = rnn.predict(padded_sequences_RNN)
        prediction_RNN = prediction_RNN.tolist()
        result_RNN = prediction_RNN[0]
        type_RNN = np.argmax(np.array(result_RNN))
        type_RNN = 0 if type_RNN == 0 else 4 if type_RNN == 2 else 2

        return jsonify({'comment': initComment, "RNN": {"type": type_RNN, 'negative_score': round(result_RNN[0]*100, 2), 'neutral_score': round(result_RNN[1]*100, 2), 'positive_score': round(result_RNN[2]*100, 2)}, })
