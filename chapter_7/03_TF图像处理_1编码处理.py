import matplotlib.pyplot as plt
import tensorflow as tf
# 报错：UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
# 需要将 mode 设置为 "rb"
# image_raw_data = tf.gfile.FastGFile("cat/cat2.jpg", 'r').read()
image_raw_data_png = tf.gfile.FastGFile("cat/cat1.png", 'rb').read()
image_raw_data_jpg = tf.gfile.FastGFile("cat/cat2.jpg", 'rb').read()

with tf.Session() as sess:

    # img_data_png = tf.image.decode_png(image_raw_data_png)
    img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
    plt.imshow(img_data_jpg.eval())
    plt.show()

    # img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)
    encoded_image = tf.image.encode_jpeg(img_data_jpg)
    with tf.gfile.GFile("cat/output.jpg", "wb") as f:
        # 耗时的操作
        f.write(encoded_image.eval())
