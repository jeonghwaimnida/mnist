# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# %% data 불러오기
import sys, os
sys.path.append("C:/Users/ljh/.spyder-py3/deep-learning-from-scratch") 
import numpy as np
import pickle
from dataset.mnist import load_mnist
import matplotlib.pylab as plt

(train_image_data, train_label_data), (test_image_data, test_label_data) = load_mnist(flatten = True, normalize = False)

print(train_image_data.shape)
print(train_label_data.shape)
print(test_image_data.shape)
print(test_label_data.shape)

import matplotlib.pylab as plt
def mnist_show(n) :
    image = train_image_data[n]
    image_reshaped = image.reshape(28, 28)
    image_reshaped.shape
    label = train_label_data[n]
    plt.figure(figsize = (4, 4))
    plt.title("sample of " + str(label))
    plt.imshow(image_reshaped, cmap="gray")
    plt.show()
    
# mnist_show(2747)
mnist_show(3000)
# %% 코드적용

def sigmoid(x) : # sigmoid 함수
    return 1 / (1 + np.exp(-x))

def softmax(matrix) : # softmax 함수
    maximum_of_matrix = np.max(matrix)
    difference_from_maximum = matrix - maximum_of_matrix
    exponential_of_difference = np.exp(difference_from_maximum)
    sum_of_exponential = np.sum(exponential_of_difference)
    y = exponential_of_difference / sum_of_exponential
    return y

def get_data(): # mnist 데이터를 불러옴. 여기서는 이 중에 test 변수만을 사용할 것이다.
    (image_train, label_train), (image_test, label_test) = load_mnist(flatten=True, normalize=False)
    return image_test, label_test

def init_network() : # sample_weight 를 불러와서 신경망 구성
    with open('C:/Users/ljh/.spyder-py3/deep-learning-from-scratch/dataset/sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x) : # 테스트 케이스들을 테스트
    # hidden data 2개
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y

images, labels = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(images)) : # 각 테스트 케이스들에 대해
    y = predict(network, images[i]) #실행 결과 output 10개가 나온다
 					#각 0~9 별로 비슷 정도에 대한 수치이다
    p = np.argmax(y) # 가장 가능성이 높은(값이 큰) 것을 선택
    if p == labels[i] : # 실제 값과 비교하여, 예측과 실제가 맞으면 카운트
        accuracy_cnt += 1
    
print("Accuracy: " + str(float(accuracy_cnt) / len(images)))


