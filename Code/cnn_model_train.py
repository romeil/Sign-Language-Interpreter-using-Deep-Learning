import numpy as np
import pickle
import cv2, os
from glob import glob
from keras import optimizers
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.layers import Dropout
from keras.api.layers import Flatten
from keras.api.layers import Conv2D
from keras.api.layers import MaxPooling2D
from keras.api.callbacks import ModelCheckpoint
from keras import backend as K
import keras as s
s.backend.set_image_data_format('channels_first')
from keras import utils as np_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
	img = cv2.imread('gestures/1/1.jpg', 0)
	return img.shape

def get_num_of_classes():
	return len(glob('gestures/*'))

image_x, image_y = get_image_size()
print(f"{image_x}, {image_y}")

def cnn_model():
	num_of_classes = get_num_of_classes()
	model = Sequential()
	model.add(Conv2D(16, (2, 2), input_shape=(image_x, image_y, 1), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_of_classes, activation='softmax'))

	sgd = optimizers.SGD(learning_rate=1e-2)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	filepath="cnn_model_keras2.keras"
	checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint1]

	return model, callbacks_list

def train():
	with open("train_images", "rb") as f:
		train_images = np.array(pickle.load(f))
	with open("train_labels", "rb") as f:
		train_labels = np.array(pickle.load(f), dtype=np.int32)

	with open("val_images", "rb") as f:
		val_images = np.array(pickle.load(f))
	with open("val_labels", "rb") as f:
		val_labels = np.array(pickle.load(f), dtype=np.int32)

	train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
	val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
	train_labels = np_utils.to_categorical(train_labels, num_classes=get_num_of_classes())
	val_labels = np_utils.to_categorical(val_labels, num_classes=get_num_of_classes())

	model, callbacks_list = cnn_model()
	model.summary()
	model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=15, batch_size=500, callbacks=callbacks_list)

	scores = model.evaluate(val_images, val_labels, verbose=0)
	print("CNN Error: %.2f%%" % (100-scores[1]*100))
	model.save('cnn_model_keras2.h5')

train()
K.clear_session()
