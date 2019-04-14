import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import random

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 识别准确度
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})# 根据测试集中的图片预测数值
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))# 讲预测值与测试集中的正确数值相比较，相同为True，不同为False
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))# 将布尔类型转化为单精度浮点类型
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})# 运行accuracy，得到识别准确度
    return result

# 定义权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 生成随机数
    return tf.Variable(initial)


# 定义偏值
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # biases最好定义为正数
    return tf.Variable(initial)


# 定义卷积层
def conv2d(x, W):  # 其中的x即为输入数据（例如RGB图片，W即为上面定义的权重）
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # tensorflow中有定义好的卷积层,直接调用即可
    # 其中，strides([1,x_movements,y_movements,1])为卷积核滑动步长,padding为补零方法


# 定义池化层
def max_pool_2x2(x):  # 其中的x为convd2中运算完毕的x
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 此处调用的是tensorflow中定义好的最大池化层
    # 其中，ksize为池化核的大小，strides([1,x_movements,y_movements,1])为卷积核滑动步长,padding为补零方法


# 定义神经网络的placeholder
xs = tf.placeholder(tf.float32, [None, 784])  # 因为输入数据是28*28的矩阵(包括了所有的输入图片)
ys = tf.placeholder(tf.float32, [None, 10])   # 其中的10是因为有0—9十个数字
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs, [-1, 28, 28, 1])  # 其中的-1代表着一共输入了多少数据，两个28即为输入数据矩阵的大小，1为输入图片的通道数（因为是灰色照片）

# 定义conv1 layer
W_conv1 = weight_variable([5, 5, 1, 32])  # 其中的两个5代表卷积核的大小，1代表输入的一张图片（包含许多特征）的通道数，32代表经过这一神经层处理后生成的通道数（特征数）
b_conv1 = bias_variable([32])  # 因为生成了32个通道
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 经过卷积运算和激活函数运算得到的数据，尺寸：28*28*32
h_pool1 = max_pool_2x2(h_conv1)  # 经过最大池化运算得到的数据，尺寸：14*14*32（池化不改变通道数只改变大小）

# 定义conv2 layer
W_conv2 = weight_variable([5, 5, 32, 64])  # 其中的两个5代表卷积核的大小，32代表输入图片（包含许多特征）经过conv1 layer处理后的通道数，64代表经过这一神经层处理后生成的通道数（特征数）
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 经过卷积运算和激励函数运算得到的数据，尺寸：14*14*64
h_pool2 = max_pool_2x2(h_conv2)  # 经过最大池化运算得到的数据，尺寸：7*7*64（池化不改变通道数只改变大小）

# 定义func1 layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_falt = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # 将h_pool2的形状改为[-1,7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_falt, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 防止过拟合

# 定义func2 layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # 进行分类，计算概率

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    # 加载训练好的参数数据
    saver.restore(sess, 'Mnist_parameter_CNN/save_parameter.ckpt')

    # 通过随即数模块生成随机测试区间
    min = random.randint(0, len(mnist.test.labels))
    max = random.randint(0, len(mnist.test.labels))
    if min > max:
        min, max = max, min
    max += 1

    # 预测值与实际值的测试
    for i in range(min, max):
        # 预测值
        prediction_num = sess.run(prediction, feed_dict={xs: np.array([mnist.test.images[i]]), keep_prob: 1})
        prediction_number = np.argmax(prediction_num, axis=1)
        print('The predictive number is ', prediction_number[0], end=' ')
        # 实际值
        ys_num = sess.run(ys, feed_dict={ys: np.array([mnist.test.labels[i]]), keep_prob: 1})
        ys_number = np.argmax(ys_num, axis=1)
        print('    The actual number is ', ys_number[0])

    # 输出随机测试区间
    max -= 1
    print('The random test interval is [', min, ',', max, ']')

    # 输出该模型的识别准确度
    recognition_accuracy = compute_accuracy(mnist.test.images, mnist.test.labels)
    print('The current recognition accuracy is', recognition_accuracy)

