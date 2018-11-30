"""
 使用tf.QueueRunner tf.Coordinator 来管理多线程队列操作
"""

import tensorflow as tf

# 先申明队列
queue = tf.FIFOQueue(100, "float")
# 定义队列的入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])

# 创建多个线程来运行入队操作
# def __init__(self, queue=None, enqueue_ops=None, close_op=None,
#                cancel_op=None, queue_closed_exception_types=None,
#                queue_runner_def=None, import_scope=None):
# 表示创建了5个线程，每个线程中运行的是enqueue_op操作
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)

# 加入到TensorFlow的计算图中  使用TensorFlow的默认计算图
# def add_queue_runner(qr, collection=ops.GraphKeys.QUEUE_RUNNERS):
tf.train.add_queue_runner(qr)
# 定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    # 使用tf.train.Coordinator来协同启动线程
    coord = tf.train.Coordinator()
    # 使用tf.trainQueueRunner 时，需要明确调用tf.train.start_queue_runners来启动所有线程。否则会因为没有线程运行入队操作。
    # 当调用出队操作时，程序会一直等待入队操作被运行。（这里是理解的重点）
    # tf.train.start_queue_runners 函数会默认启动 tf.GraphKeys.QUEUE_RUNNERS集合总所有的QueueRunner。
    # 该函数只支持启动指定集合中的QueueRunner，所以在使用tf.train.add_queue_runner()和tf.train.start_queue_runners
    # 时会指定同一个集合。
    # #def start_queue_runners(sess=None, coord=None, daemon=True, start=True,
    #                         collection=ops.GraphKeys.QUEUE_RUNNERS):
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 获取队列中的取值
    for _ in range(4):
        print(sess.run(out_tensor)[0])

    # 使用tf.train.Coordinator来停止所有线程
    coord.request_stop()
    coord.join()

"""
上述程序将启动五个线程来执行队列入队的操作，其中每一个线程都将是随机数写入队列。
在每一次运行出队操作时，可以得到一个随机数。以下是其中一次运行得到的结果：
1.2411593
0.45116103
0.36767596
-0.15366153
"""


