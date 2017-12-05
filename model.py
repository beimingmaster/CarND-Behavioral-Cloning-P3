import os
import csv
import cv2
import numpy as np 

from keras.models import Sequential
from keras.layers import Flatten, Dense, MaxPool2D, Conv2D, Dropout, Lambda, Cropping2D
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

def create_nvidia_model(input_shape, output_shape, drop_out=1.0):
	print('creating nvidia model ...')
	model = Sequential()

	model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=input_shape))

	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))	

	model.add(Conv2D(filters=24, kernel_size=(5, 5), padding='valid', strides=(2, 2), activation='relu'))
	#model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(drop_out))
	
	model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='valid', strides=(2, 2), activation='relu'))
	#model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(drop_out))

	model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='valid', strides=(2, 2), activation='relu'))
	#model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(drop_out))

	model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid', strides=(1, 1), activation='relu'))
	#model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(drop_out))

	model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid', strides=(1, 1), activation='relu'))
	#model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(drop_out))

	model.add(Flatten())

	model.add(Dense(100, activation='relu'))
	#model.add(Dropout(drop_out))

	model.add(Dense(50, activation='relu'))
	#model.add(Dropout(drop_out))

	model.add(Dense(10, activation='relu'))
	#model.add(Dropout(drop_out))

	#model.add(Dense(10, activation='relu'))

	model.add(Dense(1))
	
	print('model summary: ', model.summary())
	
	return model

def create_vgg_model(input_shape, output_shape):
	print('creating vgg model ...')

def create_lenet_model(input_shape, output_shape):
	print('creating letnet model ...')
	
def train_model(model, X_train, y_train, batch_size, split_rate, shuffle, epochs, model_path=None):
	print('training model ...')
	model.compile(loss = 'mse', optimizer = 'adam')
	if model_path:
		checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)
		model.fit(X_train, y_train, batch_size=batch_size, validation_split=split_rate, shuffle=shuffle, epochs=epochs, callbacks=[checkpointer])
	else:
		model.fit(X_train, y_train, batch_size=batch_size, validation_split=split_rate, shuffle=shuffle, epochs=epochs)

def load_data(data_path):
	print('loading data ...')
	
	lines = []
	images = []
	measurements = []

	#reading csv file
	with open('%s/%s' % (data_path, 'driving_log.csv')) as f:
		reader = csv.reader(f)
		for line in reader:
			#print('line: ', line)
			lines.append(line)

	#parsing data
	for line in lines:
		#print('line: ', line)
		center_image_path = line[0].strip().split('/')[-1]
		left_image_path = line[1].strip().split('/')[-1]
		right_image_path = line[2].strip().split('/')[-1]

		#print('center image path: %s' % center_image_path)
		#print('left image path: %s' % left_image_path)
		#print('right image path: %s' % right_image_path)

		center_image = cv2.imread('%s/IMG/%s' % (data_path, center_image_path))
		left_image = cv2.imread('%s/IMG/%s' % (data_path, left_image_path))
		right_image = cv2.imread('%s/IMG/%s' % (data_path, center_image_path))
		
		#scale
		#print('center image size: ', center_image.shape)
		#print('left image size: ', left_image.shape)
		#print('right image size: ', right_image.shape)

		#center_image = cv2.resize(center_image, (0, 0), fx = 0.5, fy = 0.5)
		#left_image = cv2.resize(left_image, (0, 0), fx = 0.5, fy = 0.5)
		#right_image = cv2.resize(right_image, (0, 0), fx = 0.5, fy = 0.5)

		images.append(center_image)
		images.append(left_image)
		images.append(right_image)

		center_measurement = float(line[3])
		left_measurement = center_measurement + 0.2
		right_measurement = center_measurement - 0.2
		
		#if left_measurement > 1.0:
		#	left_measurement = 1.0
		
		#if right_measurement < -1.0:
		#	right_measurement = -1.0

		measurements.append(center_measurement)
		measurements.append(left_measurement)
		measurements.append(right_measurement)
	
	print('count of images: ', len(images))	
	print('count of measurements: ', len(measurements))
	
	return images, measurements

def main():
	#images, measurements = load_data('./my_data/20171203')
	images, measurements = load_data('./data')

	X_train = np.array(images)
	y_train = np.array(measurements)

	#data augmentation
	aug_x = []
	aug_y = []
	size = len(X_train)
	for i in range(size):
		x = X_train[i]
		y = y_train[i]
		aug_x.append(np.fliplr(x))
		aug_y.append(-y)	
	
	X_train = np.vstack((X_train, np.array(aug_x)))
	y_train = np.hstack((y_train, np.array(aug_y)))

	cv2.imwrite('./output_images/source_100.jpg', X_train[100])
	cv2.imwrite('./output_images/flip_100.jpg', aug_x[100])
	cv2.imwrite('./output_images/source_1000.jpg', X_train[1000])
	cv2.imwrite('./output_images/flip_1000.jpg', aug_x[1000])
	cv2.imwrite('./output_images/source_10000.jpg', X_train[10000])
	cv2.imwrite('./output_images/flip_10000.jpg', aug_x[10000])

	print('mean: %.3f, min: %.3f, max: %.3f' % (np.mean(y_train), np.min(y_train), np.max(y_train)))
	print('size of X_train: %d, size of y_train: %d' % (len(X_train), len(y_train)))
	input_shape = X_train.shape[1:]
	output_shape = 1
	print('image shape: ', input_shape)
	model = create_nvidia_model(input_shape, output_shape, 0.5)

	plot_model(model, to_file='model.png', show_shapes=True)

	model_path = 'model.h5'
	if os.path.isfile(model_path):
		model.load_weights(model_path)
	#train_model(model, X_train, y_train, 128, 0.2, True, 20, model_path)

if __name__ == '__main__':
	main()
