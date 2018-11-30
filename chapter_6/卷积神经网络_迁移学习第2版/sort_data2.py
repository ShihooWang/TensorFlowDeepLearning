"""
 用来将原始的图像数据整理成模型需要的输入数据

 注意：本次数据较大 数据处理时间较长 在笔记本上导致蓝屏 内测不足！
"""
import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# 原始数据目录下有5组数据
INPUT_DATA = "data/flower_photos"

# 输出文件地址 整理后的图片数据通过numpy格式保存
OUTPUT_FILE = "data/flower_processed_data.npy"

# 测试数据和验证数据的比例
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

# 读取数据并将数据分割成训练数据、验证数据、测试数据
def create_image_list(sess, testing_percentage, validation_percentage):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  # 列表推导式
    is_root_dir = True

    # 初始化各个数据集
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    # 读取所有的子目录
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取一个子目录中的所有文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']  # 后缀名
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue  # 判断数组是否为空

        for file_name in file_list:
            # 读取并解析图片，将图片转换为299x299以便inception-v3模型来处理
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()
            image = tf.image.decode_jpeg(image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize_images(image, [299, 299])
            image_value = sess.run(image)

            # 随机划分数据集
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(image_value)
                validation_labels.append(current_label)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)
        current_label += 1

    # 将训练数据随机打乱
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)
    return np.asarray([training_images, training_labels,
                       validation_images, validation_labels,
                       testing_images, testing_labels])

def main():
    with tf.Session() as sess:
        processed_data = create_image_list(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
        # 通过numpy格式保存处理后的数据
        np.save(OUTPUT_FILE, processed_data)

if __name__ == '__main__':
    main()









