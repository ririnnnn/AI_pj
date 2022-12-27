import numpy as np
import cv2
import matplotlib.pyplot as plt

from os.path import exists
from flask import Flask, request, redirect, url_for, session
from flask.templating import render_template
from tensorflow.keras.models import Sequential, load_model

global model
model = load_model('trainedmodel')
    
app = Flask(__name__)
app.secret_key = 'test'

@app.route('/')
def index(): 
    session.clear()
    return render_template("index.html")

@app.route('/practice', methods=['GET'])
def practice_get():
    message = session.get('message', '')
    return render_template('practice.html', message = message)

@app.route('/practice', methods=['POST'])
def practice_post():
    pixels = request.form['pixels']
    pixels = pixels.split(',')
    input_data = np.copy(pixels).reshape(1, 28, 28, 1)
    input_data = input_data.astype('float32')
    input_data = input_data / 255
    conner_vallue = input_data[0][0][0] + input_data[0][0][27] + input_data[0][27][0] + input_data[0][27][27]
    if conner_vallue >= 2.0:
        input_data = 1 - input_data
    #center image
    col_sum = np.where(np.sum(input_data[0], axis=0) > 0)
    row_sum = np.where(np.sum(input_data[0], axis=1) > 0)
    row_shift = 14 - int((row_sum[0][0] + row_sum[0][-1]) / 2)
    col_shift = 14 - int((col_sum[0][0] + col_sum[0][-1]) / 2)
    input_data[0] = np.roll(input_data[0], row_shift, axis=0)
    input_data[0] = np.roll(input_data[0], col_shift, axis=1)
    #make predict
    result = model.predict(input_data)
    max = 0
    number = 0
    for i in range(10):
        if result[0][i] > max:
            max = result[0][i]
            number = i
    print(number)
    session['message'] = f'number = "{number}"'
    return redirect(url_for('practice_get'))

if __name__ == '__main__':
    app.run(debug=True)