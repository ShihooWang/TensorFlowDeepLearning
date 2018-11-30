import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="vv")
w = tf.Variable(0, dtype=tf.float32, name="vv")

saver = tf.train.Saver({"vv/ExponentialMovingAverage": v, "ww/ExponentialMovingAverage": w})

with tf.Session() as sess:
    saver.restore(sess, "C:/Wang_File/saver/test/model.ckpt")
    print(v.eval(), sess.run(w))
    print(v.eval(sess), sess.run(w))
