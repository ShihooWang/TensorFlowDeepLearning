import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.examples.tutorials.mnist import input_data

# 加载MNIST数据，如果指定地址没有，那么TensorFlow会自动去下载
mnist = input_data.read_data_sets("C:/Wang_File/MNIST", one_hot=True)

print("Traning data size : ", mnist.train.num_examples)
print("Validating data size : ", mnist.validation.num_examples)
print("Testing data size : ", mnist.test.num_examples)
# print("Example training data : ", mnist.train.images[0])
print("Example training data label : ", mnist.train.labels[0])

batch_size = 100

xs, ys = mnist.train.next_batch(batch_size)
# 从train的集合中选取batch_size个训练数据
print("X shape :", xs.shape)
print("Y shape :", ys.shape)
