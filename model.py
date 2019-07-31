import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.layers import   Flatten
from keras.layers import Conv2D, MaxPooling2D
import datetime as dt

class Timer():

	def __init__(self):
		self.start_dt = None

	def start(self):
		self.start_dt = dt.datetime.now()

	def stop(self):
		end_dt = dt.datetime.now()
		print('Time taken: %s' % (end_dt - self.start_dt))

class Model():
	"""A class for an building and inferencing an lstm model"""

	def __init__(self):
		self.model = Sequential()
		self.saved_models=""
		self.history=""
		self.save_name =""

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		self.model = load_model(filepath)
	def show_plot(self):
		plot_model(self.model, to_file='model.png')
	
	def build_lstm_model(self, configs):
		timer = Timer()
		timer.start()
		self.configs=configs
		self.save_name = '%s-%s-%s' % (self.configs['training']['name'],self.configs['model']['plan'], str(configs['training']['epochs']))
		self.model.add(LSTM(100, input_shape=(configs['data']['input_timesteps'], configs['data']['input_dim']), return_sequences=True))
		self.model.add(Dropout(0.2))
		self.model.add(LSTM(100, input_shape=(configs['data']['input_timesteps'], configs['data']['input_dim']), return_sequences=True))
		self.model.add(LSTM(100, input_shape=(configs['data']['input_timesteps'], configs['data']['input_dim']), return_sequences=False))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(configs['data']['num_classes'], activation="linear"))
		self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

		print('[Model] Model Compiled')
		timer.stop()
	def build_cnn_model(self, configs):
		timer = Timer()
		timer.start()
		self.configs=configs
		self.save_name = '%s-%s-%s' % (self.configs['training']['name'],self.configs['model']['plan'], str(configs['training']['epochs']))
		self.model.add(Conv2D(32, kernel_size=(3, 3),
		                 activation='relu',
		                 input_shape=(configs['data']['input_timesteps'], configs['data']['input_dim'],configs["data"]["channels"])))
		self.model.add(Conv2D(64, (3, 3), activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))
		self.model.add(Flatten())
		self.model.add(Dense(128, activation='relu'))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(configs['data']['num_classes'], activation='softmax'))

		self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

		print('[Model] Model Compiled')
		timer.stop()
	def build_customized_model(self, configs):
		timer = Timer()
		timer.start()
		self.configs=configs
		self.save_name = '%s-%s' % (self.configs['model']['plan'], str(configs['training']['epochs']))
		self.model.add(Conv2D(32, kernel_size=(3, 3),
		                 activation='relu',
		                 input_shape=(configs['data']['input_timesteps'], configs['data']['input_dim'],configs["data"]["channels"])))
		'''

			DIY Space reference from https://keras.io/layers/core/#dense

		'''
		self.model.add(Dense(configs['data']['num_classes'], activation='softmax'))

		self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

		print('[Model] Model Compiled')
		timer.stop()
	def train(self, x, y, epochs, batch_size):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		self.saved_models = '%s/%s-e%s.h5' % (self.configs['model']['save_model_dir'],self.save_name,dt.datetime.now().strftime('%d%m%Y-%H%M%S'))
		self.history = self.model.fit(
			x,
			(y),
			validation_split=0.25,
			epochs=epochs,
			batch_size=batch_size,
			verbose=1
		)
		self.model.save(self.saved_models)

		print('[Model] Training Completed. Model saved as %s' % self.saved_models)
		timer.stop()
	def predict_point_by_point(self, data):
		#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
		predicted = self.model.predict(data)
		return predicted

	def predict_sequences_multiple(self, data, window_size, prediction_len):
		#Predict sequence of 50 steps before shifting prediction run forward by 50 steps
		prediction_seqs = []
		for i in range(int(len(data)/prediction_len)):
			curr_frame = data[i*prediction_len]
			predicted = []
			for j in range(prediction_len):
				predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
				curr_frame = curr_frame[1:]
				curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
			prediction_seqs.append(predicted)
		return prediction_seqs

	def predict_sequence_full(self, data, window_size):
		#Shift the window by 1 new prediction each time, re-run predictions on new window
		curr_frame = data[0]
		predicted = []
		for i in range(len(data)):
			predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
			curr_frame = curr_frame[1:]
			curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
		return predicted

	def find_graph(self):
		import matplotlib.pyplot as plt
		# summarize history for loss
		plt.plot(self.history.history['loss'])
		plt.plot(self.history.history['val_loss'])
		plt.title(self.save_name+'model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig('%s/%s-loss.png' %(self.configs['model']['save_graph_dir'],self.save_name))
		plt.close('all') # 关闭图 0