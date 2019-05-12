import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# 识别准确度
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})# 根据测试集中的图片预测数值
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))# 预测值与测试集中的正确数值相比较，相同为True，不同为False
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))# 将布尔类型转化为单精度浮点类型
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})# 运行accuracy，得到识别准确度
    return result# 返回识别准确度


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
xs = tf.placeholder(tf.float32, [None, 784])  
ys = tf.placeholder(tf.float32, [None, 10])   
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs, [-1, 28, 28, 1])  ）

# 定义conv1 layer
W_conv1 = weight_variable([5, 5, 1, 32]) 
b_conv1 = bias_variable([32]) 
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  
h_pool1 = max_pool_2x2(h_conv1)

# 定义conv2 layer
W_conv2 = weight_variable([5, 5, 32, 64]) 
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  
h_pool2 = max_pool_2x2(h_conv2)  

# 定义func1 layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_falt = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_falt, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) 

# 定义func2 layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # 进行分类，计算概率

# 定义交叉熵损失函数（分类问题常用交叉熵损失函数）
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
# 定义训练函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())  # 激活神经网络
    counter = 10

    # 训练过程
    for step in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)  # 从训练集中一次提取100张图片进行训练
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if step % 1000 == 0:
            counter -= 1
            print('The remaining times trained is ', counter * 1000, '.')
            print('The current accuracy is ', compute_accuracy(mnist.test.images, mnist.test.labels), '.')
            print()

    # 保存神经网络中的参数
    saver = tf.train.Saver()
    save_path = saver.save(sess, 'Mnist_parameter_CNN/save_parameter.ckpt')

    # 提示神经网络已经完成训练
    print('The training has been end!')


