from flask import Flask, request, jsonify
from Analysis.singleComment import single_comment_analysis
from Analysis.RNN import get_Comment_Analysis_RNN, get_Comment_Analysis_pagination_RNN
from getComments import get_Comment_try
from flask_cors import CORS
from test import data
from flask import Flask

app = Flask(__name__)


@app.route('/')
def home():
    return 'Hello, World!'


@app.route('/about')
def about():
    return 'About'


app = Flask(__name__)
CORS(app)

model = "LSTM"


@app.route("/test", methods=['GET'])
def test_endpoint():
    youtubeLink = request.args.get('youtubeLink')
    comment = request.args.get('comment')
    model = request.args.get('model')
    pageNumber = request.args.get('pageNumber')
    return jsonify({'youtubeLink': youtubeLink,
                    "comment": comment,
                    "model": model,
                    "pageNumber": pageNumber,
                    "data": data})


@app.route('/get_comments', methods=['GET'])
def get_comments():
    return get_Comment_try()


@app.route('/get_comments_analysis', methods=['GET'])
def get_comments_Analysis():
    global model
    model = request.args.get('model')
    pageNumber = request.args.get('pageNumber')
    print(model)
    return get_Comment_Analysis_RNN()
    # return get_Comment_Analysis_GRU()


@app.route('/get_comments_analysis_pagination', methods=['GET'])
def get_comments_Analysis_pagination():
    pageNumber = request.args.get('pageNumber')
    return get_Comment_Analysis_pagination_RNN(pageNumber)


@app.route('/predict/text', methods=['GET'])
def predict_endpoint():
    return single_comment_analysis()


@app.route('/test 1')
def home_endpoint():
    return "Welcome"


@app.route('/flask')
def homes_endpoint():
    return "Welcome"


if __name__ == '_main_':
    app.run(debug=True)
