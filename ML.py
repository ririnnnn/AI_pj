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
def define_model():
	model = keras.models.Sequential()
	model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(keras.layers.MaxPooling2D((2, 2)))
	model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(keras.layers.MaxPooling2D((2, 2)))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(keras.layers.Dense(10, activation='softmax'))
	opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
 
#train model
def train(trainX, trainY, testX, testY, batch_sz, model, model_path, diag_path):
	history = model.fit(trainX, trainY, epochs=10, batch_size=batch_sz, validation_data=(testX, testY), verbose=0)
	#evaluate on test dataset
	_, acc = model.evaluate(testX, testY, verbose=0)
	print('> %.3f' % (acc * 100.0))
	#summarize diagnostic
	# plot loss
	plt.subplot(2, 1, 1)
	plt.title('Cross Entropy Loss')
	plt.plot(history.history['loss'], color='blue', label='train')
	plt.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	plt.subplot(2, 1, 2)
	plt.title('Classification Accuracy')
	plt.plot(history.history['accuracy'], color='blue', label='train')
	plt.plot(history.history['val_accuracy'], color='orange', label='test')
	# save diagnostics & model
	plt.savefig(diag_path)
	model.save(model_path)

#main program
trainX, trainY, testX, testY = load_dataset()
model = define_model()
train(trainX, trainY, testX, testY, 64, model, "trained_model" , "diag.png")
