# 通过变量实现神经网络的参数并实现向前传播的过程
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# 声明 w1、w2两个变量。通过seed参数设定了随机种子
# 这样可以保证每次运行的到的结果是一样的

w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

# 暂时将输入的特征向量定义为一个常量。 注意 这里x是一个1*2的矩阵

x = tf.constant([[0.7, 0.9]])
# 通过向前传播算法，获得神经网络的输出

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
sess = tf.Session()
# 初始化w1 w1
# sess.run(w1.initializer)
# sess.run(w2.initializer)
# print(sess.run(y))

# 也可以使用快捷方式
# init_top = tf.initialize_all_variables()
init_top = tf.global_variables_initializer()
sess.run(init_top)
result = sess.run(y)
print(result)
sess.close()


