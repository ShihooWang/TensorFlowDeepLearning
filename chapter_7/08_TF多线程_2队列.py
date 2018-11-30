import tensorflow as tf
import numpy as np
import threading
import time

# 每隔1秒 判断自己是否需要停止，并打印ID
def MyLoop(coord, worler_id):
    # 使用tf.Coordinator类提供的协同工具判断当前线程是否需要停止
    while not coord.should_stop():
        # 随机停止所有线程
        if np.random.rand() < 0.1:
            print("Stop from id: %d\n" % worler_id)
            # 调用 coord.request_stop()来通知其他线程
            coord.request_stop()
        else:
            print("Working id : %d\n" % worler_id)

        # 暂停1秒
        time.sleep(1)

# 申明一个tf.train.Coordinator类来协同多个线程

coord = tf.train.Coordinator()
# 创建5个线程
threads = [threading.Thread(target=MyLoop, args=(coord, i)) for i in range(5)]
# threads = [threading.Thread(target=MyLoop(coord, i)) for i in range(5)]
# 启动线程
for t in threads:
    t.start()
# 等待所有线程退出
coord.join(threads)
