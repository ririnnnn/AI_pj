import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from sklearn.model_selection import KFold

#load mnist dataset and nomalixed images
def load_dataset():
    (trainX, trainY), (testX, testY) = keras.datasets.mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    trainY = keras.utils.to_categorical(trainY)
    testY = keras.utils.to_categorical(testY)
    trainX = trainX.astype('float32')
    trainX = trainX / 255.0
    testX = testX.astype('float32')
    testX = testX / 255.0
    return trainX, trainY, testX, testY

# define cnn model
def define_model(layer_conf):
	model = keras.models.Sequential()
	model.add(keras.layers.Conv2D(layer_conf[0], (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	print("added layer ", layer_conf[0])
	model.add(keras.layers.MaxPooling2D((2, 2)))
	for layer in layer_conf[1:]:
		model.add(keras.layers.Conv2D(layer, (3, 3), activation='relu', kernel_initializer='he_uniform'))
		print("added layer ", layer)
	model.add(keras.layers.MaxPooling2D((2, 2)))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(keras.layers.Dense(10, activation='softmax'))
	opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

#train
def train_model(dataX, dataY, n_folds, batch_sz, model):
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	for train_ix, test_ix in kfold.split(dataX):
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		model.fit(trainX, trainY, epochs=10, batch_size=batch_sz, validation_data=(testX, testY), verbose=0)
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
	model.save("trainedmodel")

#main program
trainX, trainY, testX, testY = load_dataset()
layers = [64, 64]
model = define_model(layers)
#train_model(trainX, trainY, 5, 32, model)
model.fit(trainX, trainY, epochs=10, batch_size=64, validation_data=(testX, testY), verbose=0)
_, acc = model.evaluate(testX, testY, verbose=0)
print('> %.3f' % (acc * 100.0))
model.save("trainedmodel2")