import matplotlib.pyplot as plt
import tensorflow as tf
# 报错：UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
# 需要将 mode 设置为 "rb"
# image_raw_data = tf.gfile.FastGFile("cat/cat2.jpg", 'r').read()
# image_raw_data_png = tf.gfile.FastGFile("cat/cat1.png", 'rb').read()
image_raw_data_jpg = tf.gfile.FastGFile("cat/cat2.jpg", 'rb').read()

with tf.Session() as sess:

    img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)

    # 经过类型转换
    img_data = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)

    # adjusted = tf.image.adjust_brightness(img_data, -0.5)  # 亮度 -0.5
    # adjusted = tf.image.adjust_brightness(img_data, 0.5)   # 亮度 +0.5
    adjusted = tf.image.random_brightness(img_data, 0.5)  # 在 -0.5-0.5之间随机调整

    # 该API可能会导致像素的实数值超出 0.0 - 1.0 的范围， 这里需要将其截断在该范围内
    adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)
    plt.imshow(adjusted.eval())
    plt.show()

