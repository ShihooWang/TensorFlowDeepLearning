"""
如何使用tf.Coordinator
"""
import tensorflow as tf

# 创建一个先进先出的队列 并指定最多保存的元素个数 及元素类型
q = tf.FIFOQueue(2, dtypes="int32")
# 初始化
init = q.enqueue_many(([1, 4],))
# 第一个元素出列
x = q.dequeue()
y = x + 1
# 重新入列
q_inc = q.enqueue([y])

with tf.Session() as sess:
    init.run()
    for _ in range(5):
        v, _ = sess.run([x, q_inc])
        print(" %d " % v)
        # print(" %1d : %2d" % (v, 3))
        # print(" %d : d" % (v))
