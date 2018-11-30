import matplotlib.pyplot as plt
import tensorflow as tf
# 报错：UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
# 需要将 mode 设置为 "rb"
# image_raw_data = tf.gfile.FastGFile("cat/cat2.jpg", 'r').read()
image_raw_data_png = tf.gfile.FastGFile("cat/cat1.png", 'rb').read()
image_raw_data_jpg = tf.gfile.FastGFile("cat/cat2.jpg", 'rb').read()

with tf.Session() as sess:

    img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)

    # 经过类型转换
    img_data = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)
    # resized = tf.image.resize_images(img_data, [300, 300], method=0) # 双线性插值法
    # resized = tf.image.resize_images(img_data, [300, 300], method=1) # 最近邻居法
    # resized = tf.image.resize_images(img_data, [300, 300], method=2) # 双三次插值法
    # resized = tf.image.resize_images(img_data, [300, 300], method=3)  # 面积插值法
    resized = tf.image.resize_image_with_crop_or_pad(img_data, 300, 300)  # 裁剪图片
    # resized = tf.image.resize_image_with_crop_or_pad(img_data, 700, 700)  # 裁剪图片
    plt.imshow(resized.eval())
    plt.show()

