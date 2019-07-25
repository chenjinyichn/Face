from skimage import io, transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
import pickle

# 导入数据
data = np.load('data.npy') # (6000,128,128,3)
labels = np.load('labels.npy').reshape(6000, )

# 打乱数据
# 图片尺寸要求: 128*128，如果不是需要调整
n = len(data)
w = 128
h = 128
c = 3
arr = np.arange(n)  # start与stop指定的范围以及step设定的步长，生成一个ndarray
np.random.shuffle(arr)
data = data[arr]
labels = labels[arr]
# labels = tf.one_hot(labels, 10)

# 将所有数据分为训练集和验证集
ratio = 0.8
s = np.int(6000 * ratio)
x_train = data[:s]
y_train = labels[:s]
x_val = data[s:]
y_val = labels[s:]

# 定义一个函数，按批次取数据
def minibatches(x_data=None, y_data=None, batch_size=None, shuffle=False):
    assert len(x_data) == len(y_data)
    for start_idx in range(0, len(x_data) - batch_size + 1, batch_size):
        if shuffle:
            indices = np.arange(len(x_data))
            np.random.shuffle(indices)
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield x_data[excerpt], y_data[excerpt]

# ========== 构建CNN ==========
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

def CNNlayer():
    # 第一个卷积层（128——>64)
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 第二个卷积层(64->32)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # 第三个卷积层(32->16)
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # 第四个卷积层(16->8)
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    re1 = tf.reshape(pool4, [-1, 8 * 8 * 128])

    # 全连接层
    dense1 = tf.layers.dense(inputs=re1,
                             units=1024,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    dense2 = tf.layers.dense(inputs=dense1,
                             units=512,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    logits = tf.layers.dense(inputs=dense2,
                             units=60,
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    return logits

# ========== CNN结束 ==========

logits = CNNlayer()  # 10个0到1的数，代表着对应的概率

loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
y = tf.cast(tf.argmax(logits, 1), tf.int32)  # 改变某个张量的数据类型
correct_prediction = tf.equal(y, y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练和测试数据，可将n_epoch设置更大一些
saver = tf.train.Saver(max_to_keep=3)
max_acc = 0

epoch_n = 5
batch_size = 60

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for epoch in range(epoch_n):
    f = open('./recog_model/acc.txt', 'a')
    start_time = time.time()
    # training
    train_loss, train_acc, batch_n = 0, 0, 0
    last_acc = 0
    for x_train_bat, y_train_bat in minibatches(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_bat, y_: y_train_bat})
        train_loss += err
        train_acc += ac
        batch_n += 1
        print('epoch', epoch, 'batch', batch_n)
    print('train loss: %f' % (train_loss / batch_n))
    print('train acc: %f' % (train_acc / batch_n))

    # validation
    val_loss, val_acc, batch_n = 0, 0, 0
    for x_val_bat, y_val_bat in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x: x_val_bat, y_: y_val_bat})
        val_loss += err
        val_acc += ac
        batch_n += 1
        print('batch', batch_n)
    print('validation loss: %f' % (val_loss / batch_n))
    print('validation acc: %f' % (val_acc / batch_n))

    f.write('epoch: ' + str(epoch) + ' validation acc: ' + str((val_acc / batch_n)) + '\n')
    if val_acc > max_acc:
        max_acc = val_acc
        saver.save(sess, 'recog_model/faces.ckpt', global_step=epoch + 1)
    f.close()

sess.close()