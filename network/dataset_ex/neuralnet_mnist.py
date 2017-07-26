# coding: utf-8 
import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from mnist.mnist import load_mnist
from activation.activation_function import sigmoid, softmax

def get_data():
	(image_train, label_train), (image_test, label_test) = \
		load_mnist(normalize=True, flatten=True, one_hot_label=False)
	return image_test, label_test

def init_network(): # 이미 학습되었다고 가정한 weight, bias
	with open("sample_weight.pkl", 'rb') as f:
		network = pickle.load(f)

	return network

def predict(network, x):
	W1, W2, W3 = network['W1'], network['W2'], network['W3']
	b1, b2, b3 = network['b1'], network['b2'], network['b3']

	a1 = np.dot(x, W1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1, W2) + b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2, W3) + b3
	y = softmax(a3)

	return y

image_test, label_test = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(image_test)):
	y = predict(network, image_test[i])
	p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스
	if p == label_test[i]: # 예측한 값과 실제 레이블이 같으면
		accuracy_cnt += 1

print("Accuracy: " + str(float(accuracy_cnt) / len(image_test)))