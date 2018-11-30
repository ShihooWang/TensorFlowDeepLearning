import os.path
import glob
import random
import numpy as np
from tensorflow.python.platform import gfile
from chapter_6.卷积神经网络_迁移学习 import constant


# 定义从数据文件夹中读取所有的图片列表并按照，训练，验证，测试数据分开
def create_image_list(testing_percentage, validation_percentage):
    # 使用字典存储
    result = {}
    # 获取目录下所有的子目录
    sub_dirs = [x[0] for x in os.walk(constant.INPUT_DATA)]

    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:  # 第一个目录是当前目录
            is_root_dir = False
            continue
        # 获取当前目录下所有的有效图片文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(constant.INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        # 通过目录名获取类别的名称
        label_name = dir_name.lower()
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # 随机将数据分到训练集、测试集、和验证集
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result

# 通过类别名称、所属数据集和图片编号来获取一张图片的地址
def get_image_path(image_lists, image_dir, label_name, index, category):
    print(label_name, category)
    # 先找出给定标签的所有数据集
    label_lists = image_lists[label_name]
    # 再获取所属集合类型
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    # 获取图片的文件名
    base_name = category_list[mod_index]
    sub_dir = label_lists["dir"]
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path

# 通过类别名称、所属数据集合和图片编号，获取经过inception-v3模型处理之后的特征向量文件地址
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, constant.CACHE_DIR, label_name, index, category) + ".txt"

# 加载训练好的inception-v3模型处理一张图片，得到这个图片的特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # 经过卷积处理的结果是一个四维的数组，需要压缩成一维数组
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

# 获取一张图片经过inception-v3处理之后的特征向量
def get_or_create_bottleneck(sess, image_lists, label_name,
                             index, category, jepg_data_tensor,
                             bottleneck_tensor):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(constant.CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)

    # 如果这个特征向量文件不存在，则通过inception-v3模型来计算特征向量，并存储
    if not os.path.exists(bottleneck_path):
        # 获取原始的图片路径
        image_path = get_image_path(image_lists, constant.INPUT_DATA, label_name, index, category)
        # 获取图片内容
        image_data = gfile.FastGFile(image_path, 'rb').read()
        # 通过inception-v3模型计算特征向量
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jepg_data_tensor, bottleneck_tensor)
        # 将计算的到的特征向量存入文件
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, "w") as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        # 直接从文件中获取图片相应的特征向量
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values

# 随机获取一个batch的图片作为训练数据。
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many,
                                  category, jepg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        # 随机一个列表和图片的编号加入当前的训练数据
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65535)
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, category,
                                             jepg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths

# 获取全部的测试数据
def get_test_bottlenecks(sess, image_lists, n_classes, jepg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    # 枚举所有的类别和每个类别中 的测试图片
    for label_index, label_name in enumerate(label_name_list):
        category = "testing"
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                                  index, category, jepg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)

    return bottlenecks, ground_truths
