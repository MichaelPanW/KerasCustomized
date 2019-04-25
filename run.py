import os
from core.model import Model
import numpy as np
import json
import time
import keras
from keras.datasets import mnist
from sklearn.model_selection  import train_test_split
def run_task(configs,x_train, y_train):
	starttime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
	model = Model()
	if(configs['model']["plan"]=="cnn"):
		model.build_cnn_model(configs)
	if(configs['model']["plan"]=="lstm"):
		model.build_lstm_model(configs)
	model.train(
		x_train,
		y_train,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size']
	)
	endtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
	f = open('log.csv','a')
	f.write(configs['training']['name']+','+starttime+','+endtime+',"'+str(x_train.shape)+'",'+str(configs['training']['epochs'])+','+model.saved_models)
	f.close()
	return model
def loadModel(configs,m5Name):
	model = Model()
	if(configs['model']["plan"]=="cnn"):
		model.build_cnn_model(configs)
	if(configs['model']["plan"]=="lstm"):
		model.build_lstm_model(configs)
	model.load_model(m5Name)
	return model

def predict(model,x_test=[],y_test=[]):
	predicted = model.predict_point_by_point(x_test)
	max_pred=np.argmax(predicted, axis=1)
	arg_y=np.argmax(y_test, axis=1)
	f = open('log.csv','a')
	f.write(","+str(np.mean(arg_y == max_pred))+"\n")
	f.close()
	print(np.mean(arg_y == max_pred))

def main():
	#find mnist data set , you can change your dataset
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	configs = json.load(open('config.json', 'r'))
	#dataset pre process 
	#cnn need more 1 row
	if(configs['model']["plan"]=="cnn"):
		x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1)
		x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)
	#normalization
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	#2=>[0,0,1,0,0,0,0,0]
	y_train = keras.utils.to_categorical(y_train, configs["data"]["num_classes"])
	y_test = keras.utils.to_categorical(y_test, configs["data"]["num_classes"])
	#train your model
	final_model=run_task(configs,x_train, y_train)
	
	#final_model=loadModel(configs,"saved_models/tech_project-cnn-10-e25042019-100839.h5")
	
	predict(final_model,x_test, y_test)
	final_model.find_graph();
if __name__=='__main__':
	main()