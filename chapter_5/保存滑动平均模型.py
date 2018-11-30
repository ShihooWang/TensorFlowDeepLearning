import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="vv")
ww = tf.Variable(0, dtype=tf.float32, name="ww")

# 这里只是测试下当前模型中的变量数 输出 vv:0
for variables in tf.global_variables():
    print(variables.name)

ema = tf.train.ExponentialMovingAverage(0.99)  # 滑动平均
maintain_averages_op = ema.apply(tf.global_variables())
# 在模型声明之后 TensorFlow 会自动生成一个影子变量
# 输出 ：
# vv:0
# vv/ExponentialMovingAverage:0
for variables in tf.global_variables():
    print(variables.name)

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    # 保存时TensorFlow会将 vv:0 和 vv/ExponentialMovingAverage:0 两个变量都保存下来
    saver.save(sess, "C:/Wang_File/saver/test/model.ckpt")
    print(sess.run(v), sess.run(ema.average(v)))
    print(sess.run(ww), sess.run(ema.average(ww)))

