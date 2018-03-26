import pandas
import numpy
import tensorflow

csv = pandas.read_csv("./sample/bmi.csv")

# normalize. (adjust data to range from 0 to 1)
csv["height"] = csv["height"] / 200
csv["weight"] = csv["weight"] / 100
# to vector
binary_label = {"thin": [1, 0, 0], "normal": [0, 1, 0], "fat": [0, 0, 1]}
csv["binary_label"] = csv["label"].apply(lambda x: numpy.array(binary_label[x]))

# extract data to train
test_csv = csv[15000:20000]
test_data = test_csv[["weight", "height"]]
test_label = list(test_csv["binary_label"])

x = tensorflow.placeholder(tensorflow.float32, shape=[None, 2], name="x") # ?*2 2 dimension
y_ = tensorflow.placeholder(tensorflow.float32, shape=[None, 3], name="y_")

with tensorflow.name_scope('interface') as scope:
    W = tensorflow.Variable(tensorflow.zeros([2, 3]), name="W")  # weight [ [0, 0, 0], [0, 0, 0] ]
    b = tensorflow.Variable(tensorflow.zeros([3]), name="bias")  # bias [0, 0, 0]
    with tensorflow.name_scope('softmax') as scope:
        y = tensorflow.nn.softmax(tensorflow.matmul(x, W) + b) # matmul : Multiplies matrix

# training
with tensorflow.name_scope('loss') as scope:
    cross_entropy = - tensorflow.reduce_sum(y_ * tensorflow.log(y)) # error function
with tensorflow.name_scope('training') as scope:
    learning_rate = 0.01
    optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cross_entropy)

# prediction
with tensorflow.name_scope('accuracy') as scope:
    predict = tensorflow.equal(tensorflow.argmax(y, 1), tensorflow.argmax(y_, 1))
    # evaluation
    accuracy = tensorflow.reduce_mean(tensorflow.cast(predict, tensorflow.float32))

session = tensorflow.Session()
# TensorBoard
tw = tensorflow.train.SummaryWriter("log_dir", graph=session.graph)

session.run(tensorflow.initialize_all_variables())

# training
for step in range(3500):
    i = (step * 100) % 14000
    rows = csv[1 + i: 1 + i + 100]
    x_data = rows[["weight", "height"]]
    y_label = list(rows["binary_label"])
    feed_dictionary = {x: x_data, y_: y_label}
    session.run(train, feed_dict=feed_dictionary)
    if (step % 500 == 0):
        cross_entropy_value = session.run(cross_entropy, feed_dict=feed_dictionary)
        accuracy_rate = session.run(accuracy, feed_dict={x: test_data, y_: test_label})
        print("step : ", step, " cross entropy : ", cross_entropy_value, " accuracy rate : ", accuracy_rate)

# final accuracy
accuracy_rate = session.run(accuracy, feed_dict={x: test_data, y_: test_label})
print("final accuracy rate : ", accuracy_rate)
