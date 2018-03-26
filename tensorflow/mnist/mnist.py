import tensorflow
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("sample/", one_hot=True)

pixels = 28 * 28
nums = 10

x = tensorflow.placeholder(tensorflow.float32, shape=(None, pixels), name="x")
y_ = tensorflow.placeholder(tensorflow.float32, shape=(None, nums), name="y_")


def weight_variable(name, shape):
    W_init = tensorflow.truncated_normal(shape, stddev=0.1)
    W = tensorflow.Variable(W_init, name="W_" + name)
    return W

def bias_variable(name, size):
    b_init = tensorflow.constant(0.1, shape=[size])
    b = tensorflow.Variable(b_init, name="b_" + name)
    return b

def conv2d(x, W):
    return tensorflow.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool(x):
    return tensorflow.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

with tensorflow.name_scope('conv1') as scope:
    W_conv1 = weight_variable('conv1', [5, 5, 1, 32])
    b_conv1 = bias_variable('conv1', 32)
    x_image = tensorflow.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tensorflow.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

with tensorflow.name_scope('pool1') as scope:
    h_pool1 = max_pool(h_conv1)

with tensorflow.name_scope('conv2') as scope:
    W_conv2 = weight_variable('conv2', [5, 5, 32, 64])
    b_conv2 = bias_variable('conv2', 64)
    h_conv2 = tensorflow.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

with tensorflow.name_scope('pool2') as scope:
    h_pool2 = max_pool(h_conv2)

with tensorflow.name_scope('fully_connected') as scope:
    n = 7 * 7 * 64
    W_fc = weight_variable('fc', [n, 1024])
    b_fc = bias_variable('fc', 1024)
    h_pool2_flat = tensorflow.reshape(h_pool2, [-1, n])
    h_fc = tensorflow.nn.relu(tensorflow.matmul(h_pool2_flat, W_fc) + b_fc)

with tensorflow.name_scope('dropout') as scope:
    keep_prob = tensorflow.placeholder(tensorflow.float32)
    h_fc_drop = tensorflow.nn.dropout(h_fc, keep_prob)

with tensorflow.name_scope('readout') as scope:
    W_fc2 = weight_variable('fc2', [1024, 10])
    b_fc2 = bias_variable('fc2', 10)
    y_conv = tensorflow.nn.softmax(tensorflow.matmul(h_fc_drop, W_fc2) + b_fc2)

with tensorflow.name_scope('loss') as scope:
    cross_entropy = -tensorflow.reduce_sum(y_ * tensorflow.log(y_conv))
with tensorflow.name_scope('training') as scope:
    optimizer = tensorflow.train.AdamOptimizer(1e-4)
    train_step = optimizer.minimize(cross_entropy)

with tensorflow.name_scope('predict') as scope:
    predict_step = tensorflow.equal(tensorflow.argmax(y_conv, 1), tensorflow.argmax(y_, 1))
    accuracy_step = tensorflow.reduce_mean(tensorflow.cast(predict_step, tensorflow.float32))

def set_feed(images, labels, prob):
    return {x: images, y_:labels, keep_prob: prob}

with tensorflow.Session() as session:
    session.run(tensorflow.initialize_all_variables())
    tw = tensorflow.train.SummaryWriter('log_dir', graph=session.graph)
    test_feed = set_feed(mnist.test.images, mnist.test.labels, 1)
    for step in range(10000):
        batch = mnist.train.next_batch(50)
        feed = set_feed(batch[0], batch[1], 0.5)
        _, loss = session.run([train_step, cross_entropy], feed_dict=feed)
        if (step % 100 == 0):
            accuracy = session.run(accuracy_step, feed_dict=test_feed)
            print("step=", step, "loss=", loss, "acc=", accuracy)
    accuracy = session.run(accuracy_step, feed_dict=test_feed)
    print("accuracy rate = ", accuracy)