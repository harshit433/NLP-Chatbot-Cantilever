from flask import Flask, render_template, request, jsonify
import prediction
import json
from dataset.words import words, classes

model_path = 'Model/model.pth'
data = json.load(open('dataset/chatbot_data.json'))
input_size = len(words)
hidden_size = 8
output_size = len(classes)
model = prediction.load_model(model_path, input_size, hidden_size, output_size)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET', 'POST'])
def chat():
    msg = request.form['msg']
    input = msg
    return prediction.generate_response(input, model, words, classes, data)

    
if __name__ == '__main__':
    app.run()
    

