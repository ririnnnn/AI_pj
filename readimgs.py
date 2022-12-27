import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
# read image
def read_img(img_path, model):
    #load img
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #resize to 28x28
    res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    #center img
    #prepare input data
    input_data = np.copy(res).reshape(1, 28, 28, 1)
    input_data = input_data.astype('float32')
    input_data = input_data / 255
    #change background
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
    for i in range(10):
        if result[0][i] > max:
            max = result[0][i]
            number = i
    return number, max
#main
model = load_model('trained_model')
img_path = input()
number, pos = read_img(img_path, model)
print(number, '%.5f' % pos * 100)
