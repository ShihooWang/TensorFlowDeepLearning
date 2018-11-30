"""
用来处理没有整理的图片
"""

# Inception-v3 瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048

# Inception-v3模型中代表瓶颈层结果的张量名称。
BOTTLENECK_TENSOR_NAME = "pool_3/_reshape:0"

# 图像输入张量所对应的名称
JEPG_DATA_TENSOR_NAME = "DecodeJpeg/contents:0"

# 下载已训练好的inception-v3模型的目录
MODEL_DIR = "C:/Wang_File/chapter6/inception_dec_2015"

# 下载已训练好的inception-v3模型的名称
MODEL_FILE = "tensorflow_inception_graph.pb"

# 定义文件存放地址
CACHE_DIR = "C:/Wang_File/chapter6/temp/bottleneck"

# 图片数据文件夹
INPUT_DATA = "C:/Wang_File/chapter6/flower_photos"

# 验证数据的百分比
VALIDATION_PERCENTAGE = 10
# 测试数据的百分比
TEST_PERCENTAGE = 10

# 定义神经网络设置
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100

# 图片数据文件夹
SESS_DIR = "C:/Wang_File/chapter6/saver/sess"